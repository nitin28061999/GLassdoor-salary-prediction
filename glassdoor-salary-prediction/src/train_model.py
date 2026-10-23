"""Train the salary model.   Usage:  python -m src.train_model [--data PATH]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_preprocessing import (
    CAT_FEATURES, FEATURES, NUM_FEATURES, TARGET, load_and_clean,
)


def build_pipeline() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    return Pipeline([("prep", prep), ("model", rf)])


def main(data_path: str, model_path: str, metrics_path: str) -> dict:
    df = load_and_clean(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {
        "n_rows": int(len(df)),
        "mae_k": float(mean_absolute_error(y_te, pred)),
        "rmse_k": float(np.sqrt(mean_squared_error(y_te, pred))),
        "r2": float(r2_score(y_te, pred)),
        "cv_r2_mean": float(cross_val_score(build_pipeline(), X, y, cv=5, scoring="r2").mean()),
    }
    print(json.dumps(metrics, indent=2))

    # Refit on all data for the deployed model
    final = build_pipeline().fit(X, y)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path, compress=3)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {model_path}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/glassdoor_jobs.csv")
    ap.add_argument("--model", default="models/model.pkl")
    ap.add_argument("--metrics", default="models/metrics.json")
    a = ap.parse_args()
    main(a.data, a.model, a.metrics)
