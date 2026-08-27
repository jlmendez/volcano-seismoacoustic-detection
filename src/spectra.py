"""Frequency and time-frequency diagnostics for volcanic signals."""
from __future__ import annotations

import numpy as np
from scipy.signal import stft


def amplitude_spectrum(data, fs: float):
    x = np.asarray(data, dtype=float)
    window = np.hanning(len(x))
    spectrum = np.fft.rfft((x - x.mean()) * window)
    freq = np.fft.rfftfreq(len(x), d=1/fs)
    amplitude = np.abs(spectrum) * 2 / max(window.sum(), 1.0)
    return freq, amplitude


def stft_power(data, fs: float, window_s: float = 2.0, overlap: float = 0.75):
    nperseg = max(8, int(round(window_s * fs)))
    noverlap = int(round(nperseg * overlap))
    f, t, z = stft(np.asarray(data, dtype=float), fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(z) ** 2
