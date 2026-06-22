"""
TrafficIQ — Streamlit edition (Material-3 / Google-Maps inspired).
Run:  streamlit run app/main.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from action_engine import zone_stability_table, make_plan, cooldown_for  # noqa

CLEAN_CSV = os.path.join(HERE, "data", "events_clean.csv")
MODEL_DIR = os.path.join(HERE, "models")

LEVEL_COLOR = {1: "#34a853", 2: "#f9ab00", 3: "#ea8600", 4: "#ea4335"}
LEVEL_NAME = {1: "Minor", 2: "Building up", 3: "Serious", 4: "Critical"}
MAJOR = {"ORR North 1", "Bellary Road 1", "Bellary Road 2", "Tumkur Road",
         "Hosur Road", "Mysore Road", "Old Madras Road"}

st.set_page_config(page_title="TrafficIQ — Traffic Intelligence",
                   page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- MATERIAL 3 CSS ----------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #f1f3f6; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1300px; }

  /* sidebar = drawer */
  section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e3e6ea; }
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

  .brand { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
  .brand .logo { width:42px; height:42px; border-radius:12px; flex:none;
     background:linear-gradient(135deg,#1a73e8,#4285f4); color:#fff; display:grid;
     place-items:center; font-size:22px; box-shadow:0 1px 3px rgba(60,64,67,.3); }
  .brand h1 { font-size:18px; font-weight:800; margin:0; letter-spacing:-.3px; }
  .brand p { font-size:11.5px; color:#5f6368; margin:0; }
  .sect { font-size:11.5px; font-weight:700; color:#5f6368; text-transform:uppercase;
     letter-spacing:.6px; margin:16px 0 8px; }

  /* hero / stat chips */
  .hero h2 { font-size:22px; font-weight:800; margin:0 0 2px; }
  .hero p { color:#5f6368; font-size:13.5px; margin:0; }
  .chips { display:flex; gap:12px; }
  .chip { background:#fff; border:1px solid #e3e6ea; border-radius:16px; padding:12px 18px;
     box-shadow:0 1px 2px rgba(60,64,67,.2); min-width:110px; }
  .chip .v { font-size:20px; font-weight:800; }
  .chip .l { font-size:10.5px; color:#5f6368; text-transform:uppercase; letter-spacing:.3px; }

  /* result card */
  .rc { background:#fff; border-radius:24px; box-shadow:0 4px 8px rgba(60,64,67,.15),0 1px 3px rgba(60,64,67,.3);
     overflow:hidden; }
  .rc-ban { padding:20px 24px; color:#fff; }
  .rc-ban .lv { font-size:12.5px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; opacity:.92; }
  .rc-ban h2 { font-size:23px; font-weight:800; margin:2px 0 0; line-height:1.15; }
  .rc-ban .sub { font-size:13px; opacity:.92; margin-top:3px; }
  .rc-body { padding:18px 24px 22px; }
  .tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:14px; }
  .tile { background:#f8f9fc; border-radius:14px; padding:13px 14px; }
  .tile .lab { font-size:10.5px; color:#5f6368; text-transform:uppercase; letter-spacing:.3px; }
  .tile .val { font-size:21px; font-weight:800; margin-top:3px; }
  .tile .sub { font-size:10.5px; color:#80868b; }
  .money { background:#e6f4ea; border:1px solid #ceead6; color:#137333; border-radius:14px;
     padding:13px 16px; font-size:14.5px; }
  .money b { color:#0d652d; }
  .planh { font-size:12px; font-weight:700; color:#5f6368; text-transform:uppercase;
     letter-spacing:.4px; margin:18px 0 8px; }
  .barr { display:flex; align-items:center; gap:10px; background:#f8f9fc; border-left:3px solid #ea4335;
     border-radius:10px; padding:11px 14px; margin:7px 0; font-size:14px; }
  .barr b { color:#c5221f; margin-left:auto; }
  .divert { background:#e8f0fe; border:1px solid #d2e3fc; color:#1557b0; border-radius:12px;
     padding:12px 16px; font-size:13.5px; margin-top:8px; }

  /* legend */
  .legend { background:rgba(255,255,255,.9); border:1px solid #e3e6ea; border-radius:14px;
     padding:10px 14px; display:inline-block; box-shadow:0 1px 2px rgba(60,64,67,.2); }
  .legend .t { font-size:11px; font-weight:700; color:#5f6368; text-transform:uppercase; letter-spacing:.4px; }
  .legend .bar { width:220px; height:8px; border-radius:999px; margin:6px 0 4px;
     background:linear-gradient(90deg,#86efac,#fde047,#f9ab00,#ea4335,#c5221f); }
  .legend .sc { display:flex; justify-content:space-between; font-size:10px; color:#5f6368; }

  /* buttons */
  .stButton button { border-radius:12px; border:1px solid #e3e6ea; background:#f8f9fc;
     font-weight:600; transition:all .15s; }
  .stButton button:hover { border-color:#1a73e8; color:#1a73e8; }
  div[data-testid="stSidebar"] .stButton button { width:100%; }
  .predict button { background:#1a73e8 !important; color:#fff !important; border:none !important;
     border-radius:999px !important; padding:.6rem !important; font-weight:700 !important; }
  .predict button:hover { background:#1557b0 !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_CSV, low_memory=False)


@st.cache_resource
def load_models():
    reg = joblib.load(os.path.join(MODEL_DIR, "duration_model.joblib"))
    clf = joblib.load(os.path.join(MODEL_DIR, "level_model.joblib"))
    enc = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return reg, clf, enc, metrics


df = load_data()
reg, clf, enc, metrics = load_models()
stab = zone_stability_table(df)

PRESETS = {
    "🎤 Concert / Public Event": dict(etype="planned", cause="public_event",
        corridor="Bellary Road 1", zone="North Zone 2", veh="bmtc_bus",
        priority="High", hour=21, dow=5, closure=True),
    "🛕 Festival Procession": dict(etype="planned", cause="procession",
        corridor="Mysore Road", zone="West Zone 1", veh="bmtc_bus",
        priority="High", hour=18, dow=6, closure=True),
    "🚓 VIP Movement": dict(etype="planned", cause="vip_movement",
        corridor="Hosur Road", zone="South Zone 2", veh="private_car",
        priority="High", hour=10, dow=2, closure=True),
    "🚚 Truck Breakdown": dict(etype="unplanned", cause="vehicle_breakdown",
        corridor="Tumkur Road", zone="North Zone 1", veh="heavy_vehicle",
        priority="High", hour=9, dow=1, closure=False),
}

zones = sorted([z for z in df.zone.unique() if z != "Unknown"])
_def = dict(etype="unplanned", cause="public_event", corridor="Non-corridor",
            zone="North Zone 2", veh="bmtc_bus", priority="High",
            hour=21, dow=5, closure=True)
for k, v in _def.items():
    st.session_state.setdefault("in_" + k, v)
st.session_state.setdefault("go", False)


def predict(event):
    fmap = enc["freq_maps"]
    row = {
        "event_type": event["etype"], "event_cause": event["cause"],
        "veh_type": event["veh"], "corridor": event["corridor"],
        "priority": event["priority"], "hour": event["hour"],
        "day_of_week": event["dow"],
        "is_weekend": 1 if event["dow"] >= 5 else 0,
        "is_peak": 1 if event["hour"] in [7, 8, 9, 10, 17, 18, 19, 20, 21] else 0,
        "road_closure": 1 if event["closure"] else 0,
        "corridor_importance": 1.6 if event["corridor"] in MAJOR
        else (1.0 if event["corridor"] != "Non-corridor" else 0.7),
        "is_event": 1 if event["cause"] in
        {"public_event", "procession", "vip_movement", "protest"} else 0,
        "zone_freq": fmap["zone"].get(event["zone"], 0.0),
        "corridor_freq": fmap["corridor"].get(event["corridor"], 0.0),
    }
    X = pd.DataFrame([row])
    X[enc["cat"]] = enc["encoder"].transform(X[enc["cat"]].astype(str))
    dur = float(max(0, reg.predict(X[enc["reg_features"]])[0]))
    X["pred_duration"] = dur
    level = int(clf.predict(X[enc["clf_features"]])[0])
    dur = max(dur, {1: 0.3, 2: 0.6, 3: 1.2, 4: 2.0}.get(level, 0.5))
    return level, dur


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state["in_" + k] = v
    st.session_state["go"] = True


# ============================ SIDEBAR (DRAWER) ===========================
with st.sidebar:
    st.markdown("<div class='brand'><div class='logo'>🚦</div>"
                "<div><h1>TrafficIQ</h1><p>Event-Driven Traffic Intelligence · Bengaluru</p>"
                "</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='sect'>Quick scenarios</div>", unsafe_allow_html=True)
    pc = st.columns(2)
    names = list(PRESETS)
    for i, name in enumerate(names):
        pc[i % 2].button(name, key="p_" + name, on_click=apply_preset,
                         args=(name,), use_container_width=True)

    st.markdown("<div class='sect'>Plan an event</div>", unsafe_allow_html=True)
    causes = sorted(df.event_cause.value_counts().head(15).index.tolist())
    corridors = ["Non-corridor"] + sorted(
        df[df.corridor != "Non-corridor"].corridor.value_counts().head(12).index.tolist())
    vehs = sorted(df.veh_type.value_counts().head(10).index.tolist())

    st.selectbox("Event type", ["unplanned", "planned"], key="in_etype")
    st.selectbox("Cause", causes, key="in_cause")
    st.selectbox("Corridor / road", corridors, key="in_corridor")
    st.selectbox("Zone", zones, key="in_zone")
    st.selectbox("Vehicle type", vehs, key="in_veh")
    st.radio("Priority", ["High", "Low"], horizontal=True, key="in_priority")
    cc = st.columns(2)
    cc[0].slider("Hour", 0, 23, key="in_hour")
    cc[1].slider("Day (0=Mon)", 0, 6, key="in_dow")
    st.checkbox("Requires road closure?", key="in_closure")
    st.markdown("<div class='predict'>", unsafe_allow_html=True)
    if st.button("🔮  Predict & Plan", use_container_width=True):
        st.session_state["go"] = True
    st.markdown("</div>", unsafe_allow_html=True)

# ============================ MAIN ======================================
top = st.columns([2.4, 1, 1, 1])
with top[0]:
    st.markdown("<div class='hero'><h2>🚦 TrafficIQ</h2>"
                "<p>Predict congestion → grade 1–4 → auto-plan barricades, officers & diversions</p>"
                "</div>", unsafe_allow_html=True)
top[1].markdown(f"<div class='chip'><div class='v'>{len(df):,}</div>"
                "<div class='l'>Events</div></div>", unsafe_allow_html=True)
top[2].markdown(f"<div class='chip'><div class='v'>{metrics['level']['accuracy']*100:.0f}%</div>"
                "<div class='l'>Accuracy</div></div>", unsafe_allow_html=True)
top[3].markdown(f"<div class='chip'><div class='v' style='color:#ea4335'>675</div>"
                "<div class='l'>Road closures</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
map_col, side_col = st.columns([1.5, 1.05], gap="large")

# ---- HEATMAP ----
with map_col:
    lv = st.multiselect("Show congestion levels", [1, 2, 3, 4], default=[1, 2, 3, 4],
                        format_func=lambda x: f"Level {x} ({LEVEL_NAME[x]})")
    d = df[df.congestion_level.isin(lv)].dropna(subset=["latitude", "longitude"]).copy()
    d = d[["latitude", "longitude", "impact_score"]].rename(
        columns={"latitude": "lat", "longitude": "lon"})

    layer = pdk.Layer(
        "HeatmapLayer", data=d, get_position=["lon", "lat"],
        get_weight="impact_score", radius_pixels=45, intensity=1.1, threshold=0.04,
        color_range=[[134, 239, 172], [253, 224, 71], [249, 171, 0],
                     [234, 67, 53], [197, 34, 31], [127, 29, 29]],
        aggregation="SUM",
    )
    view = pdk.ViewState(latitude=12.97, longitude=77.59, zoom=10.3)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                             map_provider="carto", map_style="light"),
                    use_container_width=True, height=470)
    st.markdown("<div class='legend'><div class='t'>Traffic density</div>"
                "<div class='bar'></div><div class='sc'><span>Low</span>"
                "<span>Moderate</span><span>Severe</span></div></div>",
                unsafe_allow_html=True)

# ---- RESULT CARD ----
with side_col:
    if st.session_state["go"]:
        ev = {k: st.session_state["in_" + k] for k in _def}
        level, dur = predict(ev)
        col = LEVEL_COLOR[level]
        veh_hours = int(dur * 600 * level)
        rupees = veh_hours * 120
        cd = cooldown_for(level, 1 if ev["closure"] else 0)
        plan = make_plan({"level": level, "corridor": ev["corridor"],
                          "zone": ev["zone"], "road_closure": 1 if ev["closure"] else 0}, stab)

        barr = "".join(
            f"<div class='barr'>🚧 {b['point']} <b>{b['officers']} officers</b></div>"
            for b in plan["barricades"])
        html = f"""
        <div class='rc'>
          <div class='rc-ban' style='background:linear-gradient(135deg,{col},{col}cc)'>
            <div class='lv'>Congestion Level {level}</div>
            <h2>{LEVEL_NAME[level]}</h2>
            <div class='sub'>{ev['cause'].replace('_',' ')} on {ev['corridor']}</div>
          </div>
          <div class='rc-body'>
            <div class='tiles'>
              <div class='tile'><div class='lab'>Block time</div><div class='val'>{dur:.1f}h</div><div class='sub'>predicted</div></div>
              <div class='tile'><div class='lab'>Veh-hours</div><div class='val'>{veh_hours:,}</div><div class='sub'>lost</div></div>
              <div class='tile'><div class='lab'>Cooldown</div><div class='val'>{cd}m</div><div class='sub'>hold flow</div></div>
            </div>
            <div class='money'>💰 <b>₹{rupees:,}</b> estimated economic impact</div>
            <div class='planh'>Recommended action plan</div>
            {barr}
            <div class='divert'>🔀 Divert to: {', '.join(plan['divert_to'])} · {plan['total_officers']} officers total</div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        with st.expander("🔍 Why this plan? (explainable AI)"):
            for r in plan["reasons"]:
                st.write("•", r)
    else:
        st.info("👈 Pick a **scenario** or fill the form in the sidebar, then "
                "**Predict & Plan** to see the congestion level, ₹ impact and "
                "the barricade / officer / diversion plan here.")

# ---- ANALYTICS ----
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊  Analytics & Model insights", expanded=False):
    import plotly.express as px
    a, b, c, e = st.columns(4)
    a.metric("Level accuracy", f"{metrics['level']['accuracy']*100:.1f}%")
    b.metric("F1 (macro)", metrics['level']['f1_macro'])
    c.metric("Block-time MAE", f"{metrics['duration']['MAE_hours']} h")
    e.metric("Events analysed", f"{len(df):,}")
    g1, g2 = st.columns(2)
    with g1:
        bh = df.groupby("hour").size().reset_index(name="events")
        st.plotly_chart(px.bar(bh, x="hour", y="events", title="When disruptions happen",
                               color="events", color_continuous_scale="OrRd"),
                        use_container_width=True)
    with g2:
        wc = df[df.corridor != "Non-corridor"].corridor.value_counts().head(7).reset_index()
        wc.columns = ["corridor", "events"]
        st.plotly_chart(px.bar(wc, x="events", y="corridor", orientation="h",
                               title="Worst corridors", color="events",
                               color_continuous_scale="Reds"), use_container_width=True)
