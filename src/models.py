"""
models.py
---------
Phase 2: train the two predictors.

  Model A (regression)     : predict block_duration_hours
  Model B (classification) : predict congestion_level (1-4)

Both use the same simple input features so the dashboard can ask the user
for a few things and get predictions.

Run:  python src/models.py
Outputs: models/duration_model.joblib, models/level_model.joblib,
         models/encoders.joblib, models/metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from lightgbm import LGBMRegressor, LGBMClassifier

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(HERE, "data", "events_clean.csv")
MODEL_DIR = os.path.join(HERE, "models")

CAT_FEATURES = ["event_type", "event_cause", "veh_type", "corridor", "priority"]
NUM_FEATURES = ["hour", "day_of_week", "is_weekend", "is_peak",
                "road_closure", "corridor_importance", "is_event"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def build_xy(df):
    df = df.copy()
    for c in CAT_FEATURES:
        df[c] = df[c].fillna("unknown").astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[CAT_FEATURES] = enc.fit_transform(df[CAT_FEATURES])
    X = df[FEATURES]
    return X, enc


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN_CSV, low_memory=False)

    X, enc = build_xy(df)
    metrics = {}

    # ---------- Model A: duration (only rows that have a real duration) ----
    mask = df["block_duration_hours"].notna()
    Xr, yr = X[mask], df.loc[mask, "block_duration_hours"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = LGBMRegressor(n_estimators=400, learning_rate=0.05,
                        num_leaves=31, random_state=42, verbose=-1)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte).clip(min=0)
    metrics["duration"] = {
        "MAE_hours": round(float(mean_absolute_error(yte, pred)), 3),
        "R2": round(float(r2_score(yte, pred)), 3),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    joblib.dump(reg, os.path.join(MODEL_DIR, "duration_model.joblib"))

    # ---------- Model B: congestion level (all rows) ----------------------
    yc = df["congestion_level"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, yc, test_size=0.2, random_state=42, stratify=yc
    )
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05,
                         num_leaves=31, random_state=42, verbose=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics["level"] = {
        "accuracy": round(float(accuracy_score(yte, pred)), 3),
        "precision_macro": round(float(precision_score(yte, pred, average="macro", zero_division=0)), 3),
        "recall_macro": round(float(recall_score(yte, pred, average="macro", zero_division=0)), 3),
        "f1_macro": round(float(f1_score(yte, pred, average="macro", zero_division=0)), 3),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    joblib.dump(clf, os.path.join(MODEL_DIR, "level_model.joblib"))
    joblib.dump({"encoder": enc, "cat": CAT_FEATURES,
                 "num": NUM_FEATURES, "features": FEATURES},
                os.path.join(MODEL_DIR, "encoders.joblib"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    m = train()
    print("=== Model A: Block-time (regression) ===")
    print(f"   MAE: {m['duration']['MAE_hours']} hours   R2: {m['duration']['R2']}")
    print(f"   train/test: {m['duration']['n_train']}/{m['duration']['n_test']}")
    print("\n=== Model B: Congestion level (1-4 classification) ===")
    print(f"   Accuracy:  {m['level']['accuracy']}")
    print(f"   Precision: {m['level']['precision_macro']}  (macro)")
    print(f"   Recall:    {m['level']['recall_macro']}  (macro)")
    print(f"   F1:        {m['level']['f1_macro']}  (macro)")
    print("\nModels saved to models/. Metrics -> models/metrics.json")
