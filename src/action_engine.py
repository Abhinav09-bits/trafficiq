"""
action_engine.py
----------------
Phase 3: turn a prediction into a real action plan.

Given an event (its predicted level, corridor, zone, location) it returns:
  - barricade points         (where to put gates)
  - officers per point       (how many)
  - divert_to                (calmest nearby zone)
  - cooldown_minutes         (how long to hold/slow incoming traffic)
  - reason lines             (explainable: why this plan)

The "calm zone" ranking is learned from the data: zones with the fewest /
shortest recent disruptions are the most stable diversion targets.
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(HERE, "data", "events_clean.csv")

MAJOR_CORRIDORS = {
    "ORR East 1", "ORR East 2", "ORR North 1", "ORR North 2",
    "ORR South 1", "ORR South 2", "ORR West 1", "ORR West 2",
    "Bellary Road 1", "Bellary Road 2", "Tumkur Road", "Hosur Road",
    "Mysore Road", "Old Madras Road", "Magadi Road", "Kanakapura Road",
}


def zone_stability_table(df: pd.DataFrame) -> pd.DataFrame:
    """Lower score = calmer/more stable zone (good diversion target)."""
    g = df[df.zone != "Unknown"].groupby("zone")
    tbl = pd.DataFrame({
        "events": g.size(),
        "avg_impact": g.impact_score.mean(),
        "high_levels": g.apply(lambda x: (x.congestion_level >= 3).sum(), include_groups=False),
    })
    # normalize 0..1 and combine
    for c in ["events", "avg_impact", "high_levels"]:
        rng = tbl[c].max() - tbl[c].min()
        tbl[c + "_n"] = (tbl[c] - tbl[c].min()) / rng if rng else 0
    tbl["instability"] = (
        0.5 * tbl["events_n"] + 0.3 * tbl["avg_impact_n"] + 0.2 * tbl["high_levels_n"]
    ).round(3)
    return tbl.sort_values("instability")


def officers_for(level: int, corridor: str, road_closure: int) -> int:
    base = {1: 1, 2: 2, 3: 3, 4: 4}.get(int(level), 2)
    bonus = 0
    if corridor in MAJOR_CORRIDORS:
        bonus += 1
    if road_closure:
        bonus += 1
    return base + bonus


def cooldown_for(level: int, road_closure: int) -> int:
    """Minutes to hold/slow incoming traffic so the jam can drain."""
    base = {1: 0, 2: 10, 3: 25, 4: 45}.get(int(level), 10)
    if road_closure:
        base += 15
    return base


def make_plan(event: dict, stability: pd.DataFrame) -> dict:
    """
    event keys: level, corridor, zone, road_closure, n_barricades(optional)
    """
    level = int(event.get("level", 2))
    corridor = event.get("corridor", "Non-corridor")
    zone = event.get("zone", "Unknown")
    closure = int(event.get("road_closure", 0))

    # number of barricade points scales with severity (spread the load)
    n_points = {1: 1, 2: 2, 3: 3, 4: 4}.get(level, 2)
    per_point = officers_for(level, corridor, closure)

    # pick the calmest zones to divert toward (exclude the hotspot zone itself)
    calm = stability[stability.index != zone].head(3).index.tolist()

    barricades = []
    for i in range(n_points):
        barricades.append({
            "point": f"Approach {i+1} feeding {zone if zone!='Unknown' else corridor}",
            "officers": per_point,
        })

    total_officers = per_point * n_points

    reasons = [
        f"Predicted Level {level} -> base {{1:1,2:2,3:3,4:4}}[level] officers/point",
    ]
    if corridor in MAJOR_CORRIDORS:
        reasons.append(f"+1 officer/point: {corridor} is a major corridor")
    if closure:
        reasons.append("+1 officer/point and +15 min cooldown: road is closed")
    reasons.append(f"{n_points} barricade point(s) to SPLIT flow, not block it")
    if calm:
        reasons.append(f"Divert toward calmest zones: {', '.join(calm)}")

    return {
        "level": level,
        "barricades": barricades,
        "total_officers": total_officers,
        "cooldown_minutes": cooldown_for(level, closure),
        "divert_to": calm,
        "reasons": reasons,
    }


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    stab = zone_stability_table(df)
    print("=== Calmest zones (best diversion targets) ===")
    print(stab[["events", "avg_impact", "instability"]].head(5).round(2))
    print("\n=== Most unstable zones ===")
    print(stab[["events", "avg_impact", "instability"]].tail(5).round(2))

    demo = {"level": 4, "corridor": "Bellary Road 1",
            "zone": "North Zone 2", "road_closure": 1}
    plan = make_plan(demo, stab)
    print("\n=== SAMPLE ACTION PLAN (Level 4, Bellary Road, closed) ===")
    for b in plan["barricades"]:
        print(f"   BARRICADE: {b['point']}  -> {b['officers']} officers")
    print(f"   Total officers: {plan['total_officers']}")
    print(f"   Cooldown: {plan['cooldown_minutes']} min")
    print(f"   Divert to: {plan['divert_to']}")
    print("   Why:")
    for r in plan["reasons"]:
        print(f"     - {r}")
