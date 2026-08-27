"""K-Means based high-amplitude candidate selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["rms", "peak_abs", "kurtosis", "skewness"]


def kmeans_candidates(features: pd.DataFrame, columns=None, n_clusters: int = 3, seed: int = 42):
    columns = columns or DEFAULT_FEATURES
    x = features[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z = StandardScaler().fit_transform(x)
    model = KMeans(n_clusters=n_clusters, n_init=30, random_state=seed).fit(z)
    out = features.copy()
    out["cluster"] = model.labels_
    # candidate cluster: highest median RMS / amplitude proxy
    amplitude = out.groupby("cluster")["rms"].median()
    candidate_cluster = int(amplitude.idxmax())
    out["candidate"] = out["cluster"] == candidate_cluster
    return out, model, candidate_cluster


def merge_close_candidates(times_s, max_gap_s: float = 3.0):
    times = np.sort(np.asarray(times_s, dtype=float))
    if len(times) == 0:
        return []
    groups = [[times[0]]]
    for value in times[1:]:
        if value - groups[-1][-1] <= max_gap_s:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(float(g[0]), float(g[-1])) for g in groups]
