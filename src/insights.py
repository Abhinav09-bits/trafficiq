"""
insights.py
-----------
Phase 1: turn the clean data into (a) text insights and (b) an interactive
heatmap HTML you can open in a browser / show to judges.

Run directly:  python src/insights.py
Outputs:
  - app/heatmap.html         (interactive density map of disruptions)
  - app/hotspots_by_hour.html(bar: when disruptions happen)
"""

import os
import pandas as pd
import plotly.express as px

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(HERE, "data", "events_clean.csv")
APP_DIR = os.path.join(HERE, "app")

LEVEL_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c"}


def load():
    return pd.read_csv(CLEAN_CSV, low_memory=False)


def text_insights(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total disruptions: {len(df):,}")
    lines.append(
        f"Planned: {(df.event_type=='planned').sum():,}   "
        f"Unplanned: {(df.event_type=='unplanned').sum():,}"
    )
    lines.append(f"Road closures: {int(df.road_closure.sum()):,}")
    lines.append("")
    lines.append("Worst HOURS of day (most disruptions):")
    for h, c in df.hour.value_counts().head(5).items():
        lines.append(f"   {int(h):02d}:00  ->  {c:,} events")
    lines.append("")
    lines.append("Worst CORRIDORS:")
    for k, c in df[df.corridor != "Non-corridor"].corridor.value_counts().head(6).items():
        lines.append(f"   {k:<18} {c:,} events")
    lines.append("")
    lines.append("Highest-IMPACT causes (avg impact score):")
    top = df.groupby("event_cause").impact_score.mean().sort_values(ascending=False).head(8)
    for k, v in top.items():
        lines.append(f"   {k:<18} {v:.2f}")
    lines.append("")
    lines.append("Congestion level spread:")
    for lv, c in df.congestion_level.value_counts().sort_index().items():
        lines.append(f"   Level {lv}: {c:,}")
    return "\n".join(lines)


def make_heatmap(df: pd.DataFrame):
    os.makedirs(APP_DIR, exist_ok=True)
    d = df.dropna(subset=["latitude", "longitude"]).copy()
    d["Level"] = d["congestion_level"].astype(str)

    # 1) Density heatmap weighted by impact
    fig = px.density_mapbox(
        d, lat="latitude", lon="longitude", z="impact_score", radius=12,
        center=dict(lat=12.97, lon=77.59), zoom=10.3,
        mapbox_style="open-street-map",
        title="Bengaluru Traffic Disruption Heatmap (weighted by impact)",
        color_continuous_scale="YlOrRd",
    )
    out1 = os.path.join(APP_DIR, "heatmap.html")
    fig.write_html(out1)

    # 2) When do disruptions happen (by hour + level)
    by_hour = (
        d.groupby(["hour", "Level"]).size().reset_index(name="count")
    )
    fig2 = px.bar(
        by_hour, x="hour", y="count", color="Level",
        color_discrete_map={str(k): v for k, v in LEVEL_COLORS.items()},
        title="When disruptions happen (by hour & congestion level)",
        labels={"hour": "Hour of day", "count": "Number of events"},
    )
    out2 = os.path.join(APP_DIR, "hotspots_by_hour.html")
    fig2.write_html(out2)
    return out1, out2


if __name__ == "__main__":
    df = load()
    print(text_insights(df))
    h1, h2 = make_heatmap(df)
    print(f"\nHeatmap saved      -> {h1}")
    print(f"Hour-chart saved   -> {h2}")
    print("Open these HTML files in a browser to view.")
