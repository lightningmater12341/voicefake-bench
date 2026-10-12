"""Fixed HTK-mel MFCCs and per-clip summaries; no corpus-fitted statistics."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dct


EXTRACTOR_VERSION = "mfcc-summary-v1"


@dataclass(frozen=True)
class FeatureConfig:
    schema_version: int = 1
    sample_rate: int = 16000
    frame_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    n_mels: int = 40
    n_mfcc: int = 20
    fmin_hz: float = 20.0
    fmax_hz: float = 7600.0
    log_floor: float = 1e-12

    def __post_init__(self):
        integers = (self.schema_version, self.sample_rate, self.frame_length,
                    self.hop_length, self.n_fft, self.n_mels, self.n_mfcc)
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("Feature counts, rates, and schema must be positive integers")
        if self.schema_version != 1:
            raise ValueError("Unsupported feature configuration schema")
        if self.frame_length < 2 or self.n_fft < self.frame_length or self.hop_length > self.frame_length:
            raise ValueError("Require frame length >= 2, FFT >= frame, and hop <= frame")
        if not 1 <= self.n_mfcc <= self.n_mels <= self.n_fft // 2:
            raise ValueError("Require MFCC count <= mel count <= half the FFT size")
        values = (self.fmin_hz, self.fmax_hz, self.log_floor)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("Feature frequency and floor settings must be finite numbers")
        if not 0 <= self.fmin_hz < self.fmax_hz <= self.sample_rate / 2 or self.log_floor <= 0:
            raise ValueError("Require valid frequency bounds within Nyquist and a positive log floor")

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Feature configuration must be a JSON object")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown feature configuration fields") from exc


def mel_filterbank(config: FeatureConfig) -> np.ndarray:
    """Continuous-frequency triangles, HTK mel spacing, peak scale 1 (no area norm)."""
    mel_limits = 2595.0 * np.log10(1.0 + np.array([config.fmin_hz, config.fmax_hz]) / 700.0)
    edges = 700.0 * (10.0 ** (np.linspace(*mel_limits, config.n_mels + 2) / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(config.n_fft, 1.0 / config.sample_rate)
    rising = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    falling = (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    bank = np.maximum(0.0, np.minimum(rising, falling))
    if np.any(bank.sum(axis=1) == 0):
        raise ValueError("Empty mel filter; increase FFT resolution or reduce the mel count")
    return bank


def mfcc_frames(waveform: np.ndarray, sample_rate: int,
                config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Return complete-frame MFCCs including c0; no centering or tail padding."""
    if sample_rate != config.sample_rate:
        raise ValueError("Waveform sample rate must match the feature configuration")
    signal = np.asarray(waveform, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < config.frame_length or not np.isfinite(signal).all():
        raise ValueError("MFCC input must be a finite mono waveform at least one frame long")
    frames = np.lib.stride_tricks.sliding_window_view(signal, config.frame_length)[::config.hop_length]
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(config.frame_length) / config.frame_length)
    spectrum = np.fft.rfft(frames * window, n=config.n_fft, axis=1)
    power = np.abs(spectrum) ** 2 / config.n_fft
    energies = power @ mel_filterbank(config).T
    log_energies = np.log(np.maximum(energies, config.log_floor))
    coefficients = dct(log_energies, type=2, axis=1, norm="ortho", orthogonalize=True)[:, :config.n_mfcc]
    if not np.isfinite(coefficients).all():
        raise ValueError("MFCC computation produced non-finite values")
    return coefficients


def feature_names(config: FeatureConfig = FeatureConfig()) -> list[str]:
    return [f"mfcc_mean_{i:02d}" for i in range(config.n_mfcc)] + [
        f"mfcc_std_{i:02d}" for i in range(config.n_mfcc)]


def extract_features(waveform: np.ndarray, sample_rate: int,
                     config: FeatureConfig = FeatureConfig()) -> tuple[np.ndarray, dict]:
    frames = mfcc_frames(waveform, sample_rate, config)
    vector = np.concatenate((frames.mean(axis=0), frames.std(axis=0, ddof=0))).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Feature summary produced non-finite values")
    used = (len(frames) - 1) * config.hop_length + config.frame_length
    return vector, {"frame_count": len(frames), "unused_tail_samples": len(waveform) - used,
                    "feature_count": len(vector), "dtype": "float32"}
