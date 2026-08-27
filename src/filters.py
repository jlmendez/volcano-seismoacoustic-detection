"""Band-pass filters used by the seismic-acoustic workflow."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDS_HZ = {
    "ISA": (0.5, 20.0),   # infrasound / acoustic channel
    "IST": (8.0, 20.0),   # high-frequency seismic transient
    "SMP": (0.5, 4.0),    # low-frequency seismic component
}


def bandpass(data, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if not 0 < low < high < fs / 2:
        raise ValueError("Band edges must satisfy 0 < low < high < Nyquist")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data)


def filter_component(data, fs: float, component: str) -> np.ndarray:
    low, high = BANDS_HZ[component.upper()]
    return bandpass(data, fs, low, high)
