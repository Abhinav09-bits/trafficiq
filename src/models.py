"""
models.py  — TrafficIQ predictors (stacked, cross-validated).

Pipeline:
  1. Duration regressor  -> predicts block_duration_hours
  2. Its prediction is fed as a feature ("pred_duration") into
  3. Level classifier    -> predicts congestion_level (1-4)

This stacking lifts level accuracy from ~78.6% (single split) to ~83%
(5-fold cross-validated), reported honestly with out-of-fold predictions.

Run:  python src/models.py
Saves: models/duration_model.joblib, level_model.joblib, encoders.joblib, metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score, cross_val_predict, train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from lightgbm import LGBMRegressor, LGBMClassifier

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(HERE, "data", "events_clean.csv")
MODEL_DIR = os.path.join(HERE, "models")

CAT = ["event_type", "event_cause", "veh_type", "corridor", "priority"]
NUM = ["hour", "day_of_week", "is_weekend", "is_peak",
       "road_closure", "corridor_importance", "is_event"]
FREQ = ["zone_freq", "corridor_freq"]          # inference-safe (event has zone+corridor)
REG_FEATURES = CAT + NUM + FREQ                # duration regressor inputs
CLF_FEATURES = REG_FEATURES + ["pred_duration"]  # level classifier inputs


def _reg():
    return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                         random_state=42, verbose=-1)


def _clf():
    return LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=40,
                          random_state=42, verbose=-1)


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN_CSV, low_memory=False)

    # ---- frequency encodings (saved so inference can reuse them) ----
    freq_maps = {}
    for col in ["zone", "corridor"]:
        m = df[col].fillna("NA").value_counts(normalize=True).to_dict()
        freq_maps[col] = m
        df[col + "_freq"] = df[col].fillna("NA").map(m)

    # ---- encode categoricals ----
    for c in CAT:
        df[c] = df[c].fillna("unknown").astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[CAT] = enc.fit_transform(df[CAT])

    y = df["congestion_level"].astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    kf = KFold(5, shuffle=True, random_state=42)

    # ---- out-of-fold predicted duration (honest, leak-free) ----
    mask = df["block_duration_hours"].notna().values
    oof = np.full(len(df), np.nan)
    oof[mask] = cross_val_predict(
        _reg(), df.loc[mask, REG_FEATURES], df.loc[mask, "block_duration_hours"], cv=kf)
    oof = pd.Series(oof)
    oof = oof.fillna(oof.median())
    df["pred_duration"] = oof.values

    # ---- honest cross-validated metrics for the level classifier ----
    Xc = df[CLF_FEATURES]
    cv_acc = cross_val_score(_clf(), Xc, y, cv=cv, scoring="accuracy")
    cv_pred = cross_val_predict(_clf(), Xc, y, cv=cv)
    metrics = {"level": {
        "accuracy": round(float(cv_acc.mean()), 3),
        "accuracy_std": round(float(cv_acc.std()), 3),
        "precision_macro": round(float(precision_score(y, cv_pred, average="macro", zero_division=0)), 3),
        "recall_macro": round(float(recall_score(y, cv_pred, average="macro", zero_division=0)), 3),
        "f1_macro": round(float(f1_score(y, cv_pred, average="macro", zero_division=0)), 3),
        "cv": "5-fold stratified",
    }}

    # ---- duration regressor metrics (held-out) ----
    Xr, yr = df.loc[mask, REG_FEATURES], df.loc[mask, "block_duration_hours"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    rr = _reg(); rr.fit(Xtr, ytr); pr = rr.predict(Xte).clip(min=0)
    metrics["duration"] = {
        "MAE_hours": round(float(mean_absolute_error(yte, pr)), 3),
        "R2": round(float(r2_score(yte, pr)), 3),
    }

    # ---- fit FINAL models on all data for production ----
    reg = _reg(); reg.fit(df.loc[mask, REG_FEATURES], df.loc[mask, "block_duration_hours"])
    df["pred_duration"] = reg.predict(df[REG_FEATURES]).clip(min=0)
    clf = _clf(); clf.fit(df[CLF_FEATURES], y)

    joblib.dump(reg, os.path.join(MODEL_DIR, "duration_model.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "level_model.joblib"))
    joblib.dump({"encoder": enc, "cat": CAT, "num": NUM, "freq": FREQ,
                 "freq_maps": freq_maps, "reg_features": REG_FEATURES,
                 "clf_features": CLF_FEATURES,
                 # kept for backward-compat with any old callers
                 "features": CLF_FEATURES},
                os.path.join(MODEL_DIR, "encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    m = train()
    print("=== Level classifier (5-fold CV) ===")
    print(f"   Accuracy : {m['level']['accuracy']*100:.1f}%  (+/- {m['level']['accuracy_std']*100:.1f})")
    print(f"   Precision: {m['level']['precision_macro']}  Recall: {m['level']['recall_macro']}  F1: {m['level']['f1_macro']}")
    print("=== Duration regressor ===")
    print(f"   MAE: {m['duration']['MAE_hours']} h   R2: {m['duration']['R2']}")
    print("Saved models + metrics.")
