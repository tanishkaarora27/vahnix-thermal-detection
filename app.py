"""
VahniX -- Minimal Tactical Thermal Intelligence Dashboard (MVP)
SIH26162 -- AI-Based Detection & Classification of Industrial Fires and
Persistent Thermal Sources (NTRO).

This is a scoped-down but fully functional implementation of the pipeline
described in architecture.md. It runs on locally-generated synthetic
FIRMS-style detections (no live NASA FIRMS/OSM ingestion yet) and classifies
them with a transparent, physics-informed rule engine that mirrors the same
discriminating signals the full architecture calls for:

  - FRP surge z-score against a 90-day rolling baseline
  - Delta-NBR burn-scar collapse (flares leave the ground untouched;
    explosions destroy it)
  - Proximity to mapped industrial infrastructure

This is intentionally simple so every decision can be explained in one
sentence during a demo. Swap `generate_synthetic_detections` for a real
FIRMS/OSM ingestion call, and `classify` for the trained LightGBM + Focal
Loss ensemble + TreeSHAP, when that pipeline exists.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="VahniX -- Tactical Thermal Intelligence", layout="wide")

# ---------------------------------------------------------------------------
# Seed zones (approximate real-world coordinates, used only to place
# synthetic detections somewhere plausible on the map)
# ---------------------------------------------------------------------------
INDUSTRIAL_ZONES = [
    {"name": "Jamnagar Refinery Complex", "lat": 22.3475, "lon": 70.0797, "type": "refinery"},
    {"name": "Bokaro Steel Plant", "lat": 23.6693, "lon": 86.1511, "type": "steel"},
    {"name": "Vizag Industrial Corridor", "lat": 17.6868, "lon": 83.2185, "type": "refinery"},
    {"name": "Panipat Refinery", "lat": 29.3909, "lon": 76.9635, "type": "refinery"},
    {"name": "Rourkela Steel Plant", "lat": 22.2604, "lon": 84.8536, "type": "steel"},
]

NON_INDUSTRIAL_ZONES = [
    {"name": "Punjab Farmland Belt", "lat": 30.7333, "lon": 75.5000, "type": "agricultural"},
    {"name": "Similipal Forest Range", "lat": 21.6167, "lon": 86.2833, "type": "forest"},
    {"name": "Bandhavgarh Forest Range", "lat": 23.6000, "lon": 81.0000, "type": "forest"},
]

CLASS_COLORS = {
    "Controlled Industrial Flare": [59, 130, 246, 160],   # blue
    "Industrial Fire Emergency": [239, 68, 68, 220],      # red
    "Agricultural Stubble Burning": [234, 179, 8, 160],   # amber
    "Wildfire / Forest Fire": [249, 115, 22, 180],        # orange
    "False Positive / Noise": [148, 163, 184, 120],       # grey
}
CLASS_ORDER = list(CLASS_COLORS.keys())


@dataclass
class Detection:
    detection_id: str
    lat: float
    lon: float
    frp: float
    delta_t: float
    dist_to_industrial_km: float
    baseline_mean_frp: float
    baseline_std_frp: float
    recurrence_90d: int
    delta_nbr: float
    zone_type: str
    day_night: str


def _jitter(lat: float, lon: float, km_radius: float) -> Tuple[float, float]:
    dlat = (random.uniform(-1, 1) * km_radius) / 111.0
    dlon = (random.uniform(-1, 1) * km_radius) / (111.0 * math.cos(math.radians(lat)) + 1e-6)
    return lat + dlat, lon + dlon


def generate_synthetic_detections(n: int, seed: int) -> List[Detection]:
    rng = random.Random(seed)
    detections: List[Detection] = []

    for i in range(n):
        roll = rng.random()

        if roll < 0.55:
            # Controlled flare: stable FRP near an industrial zone
            zone = rng.choice(INDUSTRIAL_ZONES)
            lat, lon = _jitter(zone["lat"], zone["lon"], km_radius=2.0)
            baseline_mean = rng.uniform(40, 90)
            baseline_std = baseline_mean * rng.uniform(0.10, 0.30)
            frp = max(1.0, rng.gauss(baseline_mean, baseline_std * 0.6))
            delta_nbr = rng.gauss(0.0, 0.04)
            recurrence = rng.randint(45, 90)
            dist = rng.uniform(0.0, 1.2)
            zone_type = zone["type"]

        elif roll < 0.65:
            # Rare industrial emergency: FRP surge + burn scar
            zone = rng.choice(INDUSTRIAL_ZONES)
            lat, lon = _jitter(zone["lat"], zone["lon"], km_radius=1.0)
            baseline_mean = rng.uniform(40, 90)
            baseline_std = baseline_mean * rng.uniform(0.10, 0.30)
            surge_z = rng.uniform(5.0, 12.0)
            frp = baseline_mean + surge_z * baseline_std
            delta_nbr = rng.uniform(-0.75, -0.45)
            recurrence = rng.randint(0, 5)
            dist = rng.uniform(0.0, 0.6)
            zone_type = zone["type"]

        elif roll < 0.85:
            # Agricultural stubble burning
            candidates = [z for z in NON_INDUSTRIAL_ZONES if z["type"] == "agricultural"]
            zone = rng.choice(candidates or NON_INDUSTRIAL_ZONES)
            lat, lon = _jitter(zone["lat"], zone["lon"], km_radius=25.0)
            baseline_mean = rng.uniform(15, 40)
            baseline_std = baseline_mean * rng.uniform(0.3, 0.6)
            frp = max(1.0, rng.gauss(baseline_mean, baseline_std))
            delta_nbr = rng.gauss(-0.05, 0.05)
            recurrence = rng.randint(1, 10)
            dist = rng.uniform(15.0, 60.0)
            zone_type = "agricultural"

        else:
            # Wildfire / forest fire
            candidates = [z for z in NON_INDUSTRIAL_ZONES if z["type"] == "forest"]
            zone = rng.choice(candidates or NON_INDUSTRIAL_ZONES)
            lat, lon = _jitter(zone["lat"], zone["lon"], km_radius=20.0)
            baseline_mean = rng.uniform(10, 30)
            baseline_std = baseline_mean * rng.uniform(0.3, 0.6)
            frp = max(1.0, rng.gauss(baseline_mean * 1.5, baseline_std))
            delta_nbr = rng.gauss(-0.15, 0.08)
            recurrence = rng.randint(0, 6)
            dist = rng.uniform(10.0, 50.0)
            zone_type = "forest"

        delta_t = max(2.0, rng.gauss(18.0, 6.0) + (frp / 20.0))
        day_night = rng.choice(["D", "D", "N"])

        detections.append(
            Detection(
                detection_id=f"DET_{i:04d}",
                lat=lat,
                lon=lon,
                frp=round(frp, 1),
                delta_t=round(delta_t, 1),
                dist_to_industrial_km=round(dist, 2),
                baseline_mean_frp=round(baseline_mean, 1),
                baseline_std_frp=round(max(baseline_std, 0.1), 1),
                recurrence_90d=recurrence,
                delta_nbr=round(delta_nbr, 3),
                zone_type=zone_type,
                day_night=day_night,
            )
        )
    return detections


def classify(det: Detection):
    """
    Physics-informed rule-based classifier -- an MVP stand-in for the
    trained LightGBM + Focal Loss ensemble in the full architecture.
    Uses the same discriminating signals: FRP surge z-score, delta-NBR
    burn-scar collapse, and proximity to mapped industrial infrastructure.
    """
    z = (det.frp - det.baseline_mean_frp) / (det.baseline_std_frp + 1e-6)

    is_industrial_proximate = det.dist_to_industrial_km <= 2.0
    severe_burn_scar = det.delta_nbr <= -0.35
    surge = z >= 4.5

    if is_industrial_proximate and severe_burn_scar and surge:
        label = "Industrial Fire Emergency"
        risk = min(99.0, 60 + z * 3 + abs(det.delta_nbr) * 40)
        reasons = [
            (f"FRP surge z-score = {z:.1f} (>4.5 threshold)", z * 3),
            (f"delta-NBR = {det.delta_nbr:.2f} (severe structural burn scar)", abs(det.delta_nbr) * 40),
            (f"Inside industrial zone ({det.dist_to_industrial_km:.2f} km)", 15),
            (f"Recurrence (90d) = {det.recurrence_90d} (low = not routine)", 10 - det.recurrence_90d),
        ]
    elif is_industrial_proximate and not surge:
        label = "Controlled Industrial Flare"
        risk = max(2.0, 15 - z * 2)
        reasons = [
            (f"FRP close to 90-day baseline (z={z:.1f})", -z * 5),
            (f"delta-NBR ~ {det.delta_nbr:.2f} (no burn scar)", -abs(det.delta_nbr) * 10),
            (f"High recurrence ({det.recurrence_90d} passes/90d) = routine", det.recurrence_90d * 0.5),
        ]
    elif det.zone_type == "agricultural" and det.day_night == "D":
        label = "Agricultural Stubble Burning"
        risk = min(35.0, 10 + z * 2)
        reasons = [
            ("Daytime detection over farmland", 20),
            (f"Low FRP surge (z={z:.1f})", z),
            (f"Far from industrial infrastructure ({det.dist_to_industrial_km:.1f} km)", 10),
        ]
    elif det.zone_type == "forest":
        label = "Wildfire / Forest Fire"
        risk = min(70.0, 30 + z * 3)
        reasons = [
            (f"Forest canopy zone, elevated FRP surge (z={z:.1f})", z * 3),
            (f"Moderate delta-NBR ({det.delta_nbr:.2f})", abs(det.delta_nbr) * 20),
            (f"Far from industrial infrastructure ({det.dist_to_industrial_km:.1f} km)", 10),
        ]
    else:
        label = "False Positive / Noise"
        risk = max(1.0, 8 - z)
        reasons = [
            (f"No industrial proximity, no burn scar, low surge (z={z:.1f})", -z),
        ]

    reasons.sort(key=lambda r: abs(r[1]), reverse=True)
    return label, round(risk, 1), reasons


def build_dataframe(detections: List[Detection]) -> pd.DataFrame:
    rows = []
    for d in detections:
        label, risk, _ = classify(d)
        color = CLASS_COLORS.get(label, [148, 163, 184, 120])
        rows.append(
            {
                "detection_id": d.detection_id,
                "lat": d.lat,
                "lon": d.lon,
                "frp": d.frp,
                "delta_t": d.delta_t,
                "dist_to_industrial_km": d.dist_to_industrial_km,
                "recurrence_90d": d.recurrence_90d,
                "delta_nbr": d.delta_nbr,
                "zone_type": d.zone_type,
                "day_night": d.day_night,
                "predicted_class": label,
                "risk_severity_index": risk,
                "color": color,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("VahniX Control Panel")
st.sidebar.caption("SIH26162 -- NTRO Thermal Intelligence")
n_detections = st.sidebar.slider("Number of synthetic detections", 20, 300, 120, step=10)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Running on synthetic FIRMS-style detections generated locally. "
    "Swap `generate_synthetic_detections` for live NASA FIRMS/OSM ingestion "
    "when that pipeline is wired in."
)

detections = generate_synthetic_detections(int(n_detections), int(seed))
df = build_dataframe(detections)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("VahniX -- Tactical Thermal Intelligence Dashboard")
st.caption(
    "AI-Based Detection & Classification of Industrial Fires and "
    "Persistent Thermal Sources -- SIH26162 -- NTRO"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total detections", len(df))
col2.metric("Industrial emergencies", int((df["predicted_class"] == "Industrial Fire Emergency").sum()))
col3.metric("Controlled flares", int((df["predicted_class"] == "Controlled Industrial Flare").sum()))
col4.metric("Avg risk index", f"{df['risk_severity_index'].mean():.1f}")

tab_overview, tab_detail, tab_about = st.tabs(
    ["Tactical Overview", "Incident Detail", "About the Approach"]
)

with tab_overview:
    st.subheader("Live Thermal Detection Map")
    st.map(df, latitude="lat", longitude="lon", color="color", size=200)

    st.subheader("Classified Detections")
    class_filter = st.multiselect("Filter by class", CLASS_ORDER, default=CLASS_ORDER)
    filtered = df[df["predicted_class"].isin(class_filter)].sort_values(
        "risk_severity_index", ascending=False
    )
    st.dataframe(
        filtered[
            [
                "detection_id",
                "predicted_class",
                "risk_severity_index",
                "frp",
                "delta_nbr",
                "dist_to_industrial_km",
                "recurrence_90d",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_detail:
    st.subheader("Incident Diagnostic Breakdown")
    selected_id = st.selectbox("Select a detection", df["detection_id"].tolist())
    det = next(d for d in detections if d.detection_id == selected_id)
    label, risk, reasons = classify(det)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Predicted class:** {label}")
        st.markdown(f"**Risk severity index:** {risk} / 100")
        st.markdown(f"**FRP:** {det.frp} MW  |  **Delta-T:** {det.delta_t} K")
        st.markdown(f"**Distance to industrial zone:** {det.dist_to_industrial_km} km")
        st.markdown(f"**90-day recurrence:** {det.recurrence_90d} passes")
        st.markdown(f"**Delta-NBR (burn scar):** {det.delta_nbr}")

    with right:
        st.markdown("**Diagnostic factor breakdown** (feature-level attribution, MVP)")
        reason_df = pd.DataFrame(reasons, columns=["factor", "contribution"])
        st.bar_chart(reason_df.set_index("factor"))
        st.caption(
            "Lightweight rule-based attribution standing in for the TreeSHAP "
            "explainer described in the full architecture -- same "
            "discriminating physics, not yet the trained ensemble."
        )

with tab_about:
    st.markdown(
        """
### Why this looks the way it does

Standard satellite fire detection fails on industrial corridors because
mid-wave infrared brightness scales with roughly the **10th power** of
temperature -- a tiny, permanently-burning flare tip can trigger the same
alert as a genuine disaster.

This dashboard classifies each detection using three physically grounded
signals instead of a raw brightness threshold:

- **FRP surge z-score** -- how far today's radiative power is from the
  90-day rolling baseline at that location. Flares are stable; explosions
  spike.
- **Delta-NBR (burn scar index)** -- flares burn in open air and leave the
  ground untouched (`delta-NBR` near 0); explosions destroy structures
  (`delta-NBR < -0.45`).
- **Proximity to mapped industrial infrastructure** -- an OSM-style
  facility lookup, to separate industrial sources from agricultural or
  forest fires.

**What's real here:** the classification logic and the physical reasoning
behind every call the system makes.

**What's simplified for this MVP:** detections are synthetically generated
(no live NASA FIRMS feed yet), and the classifier is a transparent rule
engine rather than the trained LightGBM + Focal Loss ensemble in the full
architecture doc -- built this way on purpose so every decision can be
explained in one sentence during a demo.
        """
    )
