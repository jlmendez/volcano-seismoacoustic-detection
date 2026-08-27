"""Reusable building blocks for seismic-acoustic volcanic explosion detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Trace, read
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Band:
    low: float
    high: float
    order: int = 4


BANDS = {
    "ISA": Band(0.5, 20.0),
    "IST": Band(8.0, 20.0),
    "SMP": Band(0.5, 4.0),
}


def load_trace(path: str | Path) -> Trace:
    stream = read(str(path))
    if len(stream) != 1:
        stream.merge(method=1, fill_value="interpolate")
    trace = stream[0]
    trace.detrend("linear")
    trace.detrend("demean")
    return trace


def bandpass(data: np.ndarray, sampling_rate: float, band: Band) -> np.ndarray:
    nyquist = sampling_rate / 2.0
    if not 0 < band.low < band.high < nyquist:
        raise ValueError(f"Invalid band {band} for sampling rate {sampling_rate}")
    sos = butter(band.order, [band.low / nyquist, band.high / nyquist],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=float))


def window_features(data: np.ndarray, sampling_rate: float,
                    window_seconds: float = 2.0, step_seconds: float = 0.5) -> pd.DataFrame:
    n = int(window_seconds * sampling_rate)
    step = int(step_seconds * sampling_rate)
    rows = []
    for start in range(0, len(data) - n + 1, step):
        segment = np.asarray(data[start:start + n], dtype=float)
        rows.append({
            "start_s": start / sampling_rate,
            "rms": float(np.sqrt(np.mean(segment ** 2))),
            "peak": float(np.max(np.abs(segment))),
            "kurtosis": float(kurtosis(segment, fisher=False, bias=False)),
            "skewness": float(skew(segment, bias=False)),
        })
    return pd.DataFrame(rows)


def cluster_candidates(features: pd.DataFrame, n_clusters: int = 3,
                       random_state: int = 42) -> pd.DataFrame:
    cols = ["rms", "peak", "kurtosis", "skewness"]
    clean = features[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(clean)
    labels = KMeans(n_clusters=n_clusters, n_init=20,
                    random_state=random_state).fit_predict(scaled)
    out = features.copy()
    out["cluster"] = labels
    energetic = out.groupby("cluster")["rms"].median().idxmax()
    out["candidate"] = out["cluster"].eq(energetic)
    return out


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0, 0
    corr = np.correlate(a, b, mode="full") / norm
    index = int(np.argmax(np.abs(corr)))
    lag = index - (len(b) - 1)
    return float(corr[index]), lag


def validate_pair(seismic: np.ndarray, acoustic: np.ndarray,
                  sampling_rate: float, min_correlation: float = 0.65,
                  max_delay_s: float = 0.06) -> dict:
    correlation, lag_samples = normalized_cross_correlation(seismic, acoustic)
    delay = lag_samples / sampling_rate
    return {
        "correlation": correlation,
        "delay_s": delay,
        "validated": abs(correlation) >= min_correlation and abs(delay) < max_delay_s,
    }


def analyze_file(path: str | Path, channel_type: str) -> pd.DataFrame:
    trace = load_trace(path)
    filtered = bandpass(trace.data, trace.stats.sampling_rate, BANDS[channel_type])
    return cluster_candidates(window_features(filtered, trace.stats.sampling_rate))
