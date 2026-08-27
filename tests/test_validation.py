import numpy as np

from src.validation import normalized_cross_correlation, validate_coupling


def test_identical_signal_passes_coupling_validation():
    fs = 50.0
    t = np.arange(500) / fs
    signal = np.sin(2 * np.pi * 3.0 * t) + 0.3 * np.sin(2 * np.pi * 7.0 * t)
    result = validate_coupling(signal, signal.copy(), fs)
    assert result["passes"] is True
    assert abs(result["correlation"] - 1.0) < 1e-10
    assert abs(result["delay_s"]) < 1e-12


def test_zero_energy_signals_fail_safely():
    corr, delay = normalized_cross_correlation(np.zeros(100), np.zeros(100), fs=50.0)
    assert corr == 0.0
    assert delay == 0.0
    assert validate_coupling(np.zeros(100), np.zeros(100), fs=50.0)["passes"] is False
