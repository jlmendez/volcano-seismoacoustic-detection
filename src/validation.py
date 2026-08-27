"""Cross-correlation validation for coupled acoustic/seismic candidates."""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags


def normalized_cross_correlation(a, b, fs: float):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(len(a), len(b)); a = a[:n]; b = b[:n]
    a = a - a.mean(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0, 0.0
    corr = correlate(a, b, mode="full") / denom
    lags = correlation_lags(len(a), len(b), mode="full")
    idx = int(np.argmax(np.abs(corr)))
    return float(corr[idx]), float(lags[idx] / fs)


def validate_coupling(acoustic, seismic, fs: float, min_abs_corr: float = 0.65, max_abs_delay_s: float = 0.06) -> dict:
    corr, delay = normalized_cross_correlation(acoustic, seismic, fs)
    return {
        "correlation": corr,
        "delay_s": delay,
        "passes": bool(abs(corr) >= min_abs_corr and abs(delay) < max_abs_delay_s),
    }


def extract_centered_window(data, center_s: float, fs: float, half_width_s: float = 3.0):
    x = np.asarray(data)
    center = int(round(center_s * fs)); half = int(round(half_width_s * fs))
    lo = max(0, center-half); hi = min(len(x), center+half)
    return x[lo:hi]
