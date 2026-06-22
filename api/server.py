"""
FastAPI backend - serves the trained models + action engine to the React UI.
Run:  uvicorn api.server:app --reload --port 8000   (from traffic-system/)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from action_engine import zone_stability_table, make_plan, cooldown_for  # noqa

CLEAN_CSV = os.path.join(HERE, "data", "events_clean.csv")
MODEL_DIR = os.path.join(HERE, "models")

LEVEL_NAME = {1: "Minor", 2: "Building up", 3: "Serious", 4: "Critical"}
MAJOR_CORRIDORS = {
    "ORR North 1", "Bellary Road 1", "Bellary Road 2", "Tumkur Road",
    "Hosur Road", "Mysore Road", "Old Madras Road",
}

# ---- load once ----
df = pd.read_csv(CLEAN_CSV, low_memory=False)
reg = joblib.load(os.path.join(MODEL_DIR, "duration_model.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "level_model.joblib"))
enc = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)
STAB = zone_stability_table(df)

app = FastAPI(title="Bengaluru Traffic Intelligence API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class EventIn(BaseModel):
    etype: str = "unplanned"
    cause: str = "public_event"
    corridor: str = "Non-corridor"
    zone: str = "North Zone 2"
    veh: str = "bmtc_bus"
    priority: str = "High"
    hour: int = 21
    dow: int = 5
    closure: bool = True


def _predict(e: EventIn):
    cat = enc["cat"]
    fmap = enc["freq_maps"]
    row = {
        "event_type": e.etype, "event_cause": e.cause, "veh_type": e.veh,
        "corridor": e.corridor, "priority": e.priority, "hour": e.hour,
        "day_of_week": e.dow,
        "is_weekend": 1 if e.dow >= 5 else 0,
        "is_peak": 1 if e.hour in [7, 8, 9, 10, 17, 18, 19, 20, 21] else 0,
        "road_closure": 1 if e.closure else 0,
        "corridor_importance": 1.6 if e.corridor in MAJOR_CORRIDORS
        else (1.0 if e.corridor != "Non-corridor" else 0.7),
        "is_event": 1 if e.cause in
        {"public_event", "procession", "vip_movement", "protest"} else 0,
        "zone_freq": fmap["zone"].get(e.zone, 0.0),
        "corridor_freq": fmap["corridor"].get(e.corridor, 0.0),
    }
    X = pd.DataFrame([row])
    X[cat] = enc["encoder"].transform(X[cat].astype(str))
    # 1) predict duration, 2) feed it into the level classifier (stacked)
    dur = float(max(0, reg.predict(X[enc["reg_features"]])[0]))
    X["pred_duration"] = dur
    level = int(clf.predict(X[enc["clf_features"]])[0])
    dur = max(dur, {1: 0.3, 2: 0.6, 3: 1.2, 4: 2.0}.get(level, 0.5))
    return level, dur


@app.get("/api/options")
def options():
    causes = sorted(df.event_cause.value_counts().head(15).index.tolist())
    corridors = ["Non-corridor"] + sorted(
        df[df.corridor != "Non-corridor"].corridor.value_counts().head(12).index.tolist())
    zones = sorted([z for z in df.zone.unique() if z != "Unknown"])
    vehs = sorted(df.veh_type.value_counts().head(10).index.tolist())
    return {"causes": causes, "corridors": corridors, "zones": zones,
            "vehs": vehs, "total_events": int(len(df))}


@app.get("/api/presets")
def presets():
    return [
        {"name": "Concert / Public Event", "icon": "🎤",
         "event": {"etype": "planned", "cause": "public_event",
                   "corridor": "Bellary Road 1", "zone": "North Zone 2",
                   "veh": "bmtc_bus", "priority": "High", "hour": 21,
                   "dow": 5, "closure": True}},
        {"name": "Festival Procession", "icon": "🛕",
         "event": {"etype": "planned", "cause": "procession",
                   "corridor": "Mysore Road", "zone": "West Zone 1",
                   "veh": "bmtc_bus", "priority": "High", "hour": 18,
                   "dow": 6, "closure": True}},
        {"name": "VIP Movement", "icon": "🚓",
         "event": {"etype": "planned", "cause": "vip_movement",
                   "corridor": "Hosur Road", "zone": "South Zone 2",
                   "veh": "private_car", "priority": "High", "hour": 10,
                   "dow": 2, "closure": True}},
        {"name": "Truck Breakdown (peak)", "icon": "🚚",
         "event": {"etype": "unplanned", "cause": "vehicle_breakdown",
                   "corridor": "Tumkur Road", "zone": "North Zone 1",
                   "veh": "heavy_vehicle", "priority": "High", "hour": 9,
                   "dow": 1, "closure": False}},
    ]


@app.post("/api/predict")
def predict(e: EventIn):
    level, dur = _predict(e)
    veh_hours = int(dur * 600 * level)
    rupees = veh_hours * 120
    cd = cooldown_for(level, 1 if e.closure else 0)
    plan = make_plan({"level": level, "corridor": e.corridor, "zone": e.zone,
                      "road_closure": 1 if e.closure else 0}, STAB)
    return {
        "level": level, "level_name": LEVEL_NAME[level],
        "duration_hours": round(dur, 1),
        "vehicle_hours": veh_hours, "rupees": rupees, "cooldown_min": cd,
        "barricades": plan["barricades"], "total_officers": plan["total_officers"],
        "divert_to": plan["divert_to"], "reasons": plan["reasons"],
        "cause": e.cause, "corridor": e.corridor,
    }


@app.get("/api/heatmap")
def heatmap(levels: str = "3,4"):
    want = {int(x) for x in levels.split(",") if x.strip()}
    d = df[df.congestion_level.isin(want)].dropna(subset=["latitude", "longitude"])
    d = d[["latitude", "longitude", "impact_score", "congestion_level"]].copy()
    # cap points for performance (more points = smoother heat layer)
    if len(d) > 3500:
        d = d.sample(3500, random_state=1)
    return {"points": [
        {"lat": float(r.latitude), "lon": float(r.longitude),
         "impact": float(r.impact_score), "level": int(r.congestion_level)}
        for r in d.itertuples()]}


@app.get("/api/insights")
def insights():
    by_hour = df.groupby("hour").size().reset_index(name="events")
    corridors = (df[df.corridor != "Non-corridor"].corridor.value_counts()
                 .head(7).reset_index())
    corridors.columns = ["corridor", "events"]
    causes = (df.groupby("event_cause").impact_score.mean()
              .sort_values(ascending=False).head(7).reset_index())
    causes.columns = ["cause", "avg_impact"]
    calm = STAB[["events", "avg_impact", "instability"]].head(5).reset_index()
    return {
        "metrics": METRICS,
        "total_events": int(len(df)),
        "by_hour": by_hour.to_dict("records"),
        "corridors": corridors.to_dict("records"),
        "causes": [{"cause": c, "avg_impact": round(v, 2)}
                   for c, v in zip(causes.cause, causes.avg_impact)],
        "calm_zones": [{"zone": z, "events": int(ev), "avg_impact": round(ai, 2),
                        "instability": round(ins, 2)}
                       for z, ev, ai, ins in zip(calm.zone, calm.events,
                                                 calm.avg_impact, calm.instability)],
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "traffic-intelligence-api"}


# ---- serve the built React app (single-service deploy) ----
from fastapi.staticfiles import StaticFiles  # noqa: E402

_DIST = os.path.join(HERE, "web", "dist")
if os.path.isdir(_DIST):
    # mounted last so /api/* routes take precedence; html=True serves index.html
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
