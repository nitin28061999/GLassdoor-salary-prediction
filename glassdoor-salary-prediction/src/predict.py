"""Prediction helpers, including a tree-based uncertainty band."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_preprocessing import FEATURES


def predict_with_interval(model, row: dict, low_q: float = 10, high_q: float = 90):
    """Return (mean_k, low_k, high_k) in $K.

    For a Random Forest pipeline the band is the 10th-90th percentile of the
    individual trees' predictions. Otherwise it falls back to +/-15%.
    """
    X = pd.DataFrame([row])[FEATURES]
    mean = float(model.predict(X)[0])
    try:
        Xt = model[:-1].transform(X)
        trees = np.array([t.predict(Xt)[0] for t in model[-1].estimators_])
        return mean, float(np.percentile(trees, low_q)), float(np.percentile(trees, high_q))
    except Exception:
        return mean, mean * 0.85, mean * 1.15
