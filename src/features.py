"""Window-level features for candidate explosion detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def window_features(data, fs: float, window_s: float = 4.0, step_s: float = 1.0) -> pd.DataFrame:
    x = np.asarray(data, dtype=float)
    n = max(1, int(round(window_s * fs)))
    step = max(1, int(round(step_s * fs)))
    rows = []
    for start in range(0, max(len(x) - n + 1, 0), step):
        w = x[start:start+n]
        rows.append({
            "time_s": start / fs,
            "rms": float(np.sqrt(np.mean(w**2))),
            "peak_abs": float(np.max(np.abs(w))),
            "std": float(np.std(w)),
            "kurtosis": float(kurtosis(w, fisher=True, bias=False)),
            "skewness": float(skew(w, bias=False)),
            "energy": float(np.sum(w**2)),
        })
    return pd.DataFrame(rows)


def robust_standardize(frame: pd.DataFrame, columns) -> pd.DataFrame:
    z = frame.copy()
    for column in columns:
        median = z[column].median()
        mad = np.median(np.abs(z[column] - median)) or 1.0
        z[column] = 0.6745 * (z[column] - median) / mad
    return z
