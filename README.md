# Volcano Seismo-Acoustic Explosion Detection

A signal-processing and clustering workflow for detecting and validating volcanic explosions using seismic and acoustic/infrasound records from **Volcán de Fuego, Guatemala**.

## Highlights

- MiniSEED ingestion and waveform handling with ObsPy
- Channel-specific band-pass filtering
- Sliding-window statistical feature extraction
- K-Means candidate segmentation
- Seismic/acoustic cross-correlation validation
- Reusable functions for candidate detection and pair validation

## Tech stack

Python · ObsPy · NumPy · pandas · SciPy · scikit-learn · signal processing

## Methodological outline

1. Load and preprocess seismic or acoustic waveforms.
2. Apply channel-specific frequency bands.
3. Extract RMS, peak amplitude, kurtosis, and skewness in sliding windows.
4. Use K-Means to identify the most energetic candidate cluster.
5. Validate paired seismic/acoustic signals with normalized cross-correlation and delay constraints.

## Repository structure

- `src/seismoacoustic_detection.py` — reusable processing and detection components
- `data/external/` — location for authorized MiniSEED and catalogue files
- `requirements.txt` — Python dependencies

## Data and reproducibility

Waveform and catalogue files are intentionally not redistributed. Place authorized external data under `data/external/`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Research context

This repository distills a larger research workflow on seismic-acoustic detection at Volcán de Fuego into reusable Python components suitable for inspection, extension, and integration into monitoring pipelines.
