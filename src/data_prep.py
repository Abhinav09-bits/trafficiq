"""
data_prep.py
------------
Loads the raw Bengaluru traffic-event CSV, cleans it, and builds the
features the whole system needs:

  - block_duration_hours : how long the road was blocked (our prediction target)
  - hour / day_of_week   : when it happened
  - corridor_importance  : how big/important the road is
  - impact_score         : block_time x closure x corridor x priority x peak
  - congestion_level     : 1 (low) .. 4 (critical)  <- derived from impact_score

Run directly to produce data/events_clean.csv and print insights.
"""

import os
import numpy as np
import pandas as pd

# ---- paths -------------------------------------------------------------
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(HERE, "data", "events.csv")
CLEAN_CSV = os.path.join(HERE, "data", "events_clean.csv")

# Bengaluru rough bounding box (to drop bad GPS points)
LAT_MIN, LAT_MAX = 12.7, 13.25
LON_MIN, LON_MAX = 77.3, 77.85

# Which event causes are "real events / gatherings" vs random incidents
EVENT_CAUSES = {"public_event", "procession", "vip_movement", "protest"}

# Corridor importance weight (major ring roads / highways = higher)
MAJOR_CORRIDORS = {
    "ORR East 1", "ORR East 2", "ORR North 1", "ORR North 2",
    "ORR South 1", "ORR South 2", "ORR West 1", "ORR West 2",
    "Bellary Road 1", "Bellary Road 2", "Tumkur Road", "Hosur Road",
    "Mysore Road", "Old Madras Road", "Magadi Road", "Kanakapura Road",
}


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_and_clean(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # --- keep only valid event_type rows (data has junk in this column) ---
    df = df[df["event_type"].isin(["planned", "unplanned"])].copy()

    # --- parse the datetime columns we care about ---
    for c in ["start_datetime", "resolved_datetime", "closed_datetime"]:
        df[c] = _to_dt(df[c])

    # --- block duration = (resolved or closed) - start, in hours ---
    end = df["resolved_datetime"].fillna(df["closed_datetime"])
    dur = (end - df["start_datetime"]).dt.total_seconds() / 3600.0
    # keep sensible durations; cap extreme outliers at 24h
    dur = dur.where((dur > 0) & (dur < 72))
    dur = dur.clip(upper=24)
    df["block_duration_hours"] = dur

    # --- time features ---
    df["hour"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek  # 0=Mon
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # peak hours: morning 7-10, evening 17-21
    df["is_peak"] = df["hour"].isin([7, 8, 9, 10, 17, 18, 19, 20, 21]).astype(int)

    # --- clean GPS, drop points outside Bengaluru ---
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    good_gps = (
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
    )
    df = df[good_gps].copy()

    # --- tidy categorical fields ---
    df["event_cause"] = (
        df["event_cause"].fillna("others").astype(str).str.strip().str.lower()
    )
    df["veh_type"] = df["veh_type"].fillna("unknown").astype(str).str.strip()
    df["corridor"] = df["corridor"].fillna("Non-corridor").astype(str).str.strip()
    df["zone"] = df["zone"].fillna("Unknown").astype(str).str.strip()
    df["priority"] = df["priority"].fillna("Low").astype(str).str.strip()

    # road closure -> 0/1
    df["road_closure"] = (
        df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int)
    )

    # is this a gathering-type event?
    df["is_event"] = df["event_cause"].isin(EVENT_CAUSES).astype(int)

    # corridor importance weight
    df["corridor_importance"] = df["corridor"].apply(
        lambda c: 1.6 if c in MAJOR_CORRIDORS else (1.0 if c != "Non-corridor" else 0.7)
    )

    # priority weight
    df["priority_w"] = df["priority"].map({"High": 1.4, "Low": 1.0}).fillna(1.0)

    return df.reset_index(drop=True)


def add_impact_and_level(df: pd.DataFrame) -> pd.DataFrame:
    """Build the impact score and the 1-4 congestion level."""
    # fill missing duration with the median so scoring still works
    dur = df["block_duration_hours"].fillna(df["block_duration_hours"].median())

    closure_w = np.where(df["road_closure"] == 1, 1.8, 1.0)
    peak_w = np.where(df["is_peak"] == 1, 1.3, 1.0)

    impact = (
        dur
        * closure_w
        * df["corridor_importance"]
        * df["priority_w"]
        * peak_w
    )
    df["impact_score"] = impact.round(3)

    # turn the continuous score into 4 levels using quantile bands
    # (data-driven, so the bands fit the real distribution)
    try:
        df["congestion_level"] = (
            pd.qcut(impact, q=[0, 0.45, 0.75, 0.92, 1.0], labels=[1, 2, 3, 4])
            .astype(int)
        )
    except ValueError:
        # fallback if too many duplicate edges
        df["congestion_level"] = pd.cut(
            impact, bins=4, labels=[1, 2, 3, 4]
        ).astype(int)

    return df


def build() -> pd.DataFrame:
    df = load_and_clean()
    df = add_impact_and_level(df)
    df.to_csv(CLEAN_CSV, index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(f"Saved clean data -> {CLEAN_CSV}")
    print(f"Rows: {len(df)}   Columns: {len(df.columns)}")
    print("\n--- Congestion level counts ---")
    print(df["congestion_level"].value_counts().sort_index())
    print("\n--- Duration (hours) ---")
    print(df["block_duration_hours"].describe().round(2))
    print("\n--- Top 8 zones by event count ---")
    print(df["zone"].value_counts().head(8))
    print("\n--- Top 8 corridors ---")
    print(df["corridor"].value_counts().head(8))
    print("\n--- Avg impact by event_cause (top 10) ---")
    print(
        df.groupby("event_cause")["impact_score"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(2)
    )
