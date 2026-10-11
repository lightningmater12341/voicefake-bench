"""Deterministic waveform loading; no features, predictions, or learned transforms."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioConfig:
    schema_version: int = 1
    target_sample_rate: int = 16000
    min_duration_seconds: float = 0.25
    max_duration_seconds: float = 10.0
    max_source_duration_seconds: float = 120.0

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported audio configuration schema")
        if type(self.target_sample_rate) is not int or not 8000 <= self.target_sample_rate <= 192000:
            raise ValueError("target_sample_rate must be an integer from 8000 to 192000")
        durations = (self.min_duration_seconds, self.max_duration_seconds,
                     self.max_source_duration_seconds)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in durations):
            raise ValueError("Duration settings must be positive finite numbers")
        if not self.min_duration_seconds <= self.max_duration_seconds <= self.max_source_duration_seconds:
            raise ValueError("Require minimum <= crop maximum <= source maximum duration")

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Audio configuration must be a JSON object")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown audio configuration fields") from exc


def process_samples(samples: np.ndarray, source_rate: int,
                    config: AudioConfig = AudioConfig()) -> tuple[np.ndarray, dict]:
    """Mix channels, resample with an FIR filter, and center-crop without padding."""
    if type(source_rate) is not int or source_rate <= 0:
        raise ValueError("Source sample rate must be a positive integer")
    raw = np.asarray(samples, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("Audio must contain frames and channels")
    if not np.isfinite(raw).all():
        raise ValueError("Audio contains non-finite sample values")
    source_duration = raw.shape[0] / source_rate
    if not config.min_duration_seconds <= source_duration <= config.max_source_duration_seconds:
        raise ValueError("Source duration is outside configured limits")
    mono = raw.mean(axis=1, dtype=np.float64).astype(np.float32)
    if not np.any(mono):
        raise ValueError("Audio is zero-energy after channel mixing")
    divisor = math.gcd(source_rate, config.target_sample_rate)
    up, down = config.target_sample_rate // divisor, source_rate // divisor
    if up != down:
        mono = resample_poly(mono, up, down, window=("kaiser", 5.0),
                             padtype="constant", cval=0.0).astype(np.float32)
    before_crop = len(mono)
    limit = math.floor(config.max_duration_seconds * config.target_sample_rate)
    start = max(0, (before_crop - limit) // 2)
    stop = min(before_crop, start + limit)
    mono = np.ascontiguousarray(mono[start:stop], dtype=np.float32)
    if len(mono) < math.ceil(config.min_duration_seconds * config.target_sample_rate):
        raise ValueError("Processed audio is shorter than the configured minimum")
    if not np.isfinite(mono).all() or not np.any(mono):
        raise ValueError("Processed audio has non-finite samples or zero energy")
    details = {
        "source_sample_rate": source_rate, "source_frames": int(raw.shape[0]),
        "source_channels": int(raw.shape[1]), "source_duration_seconds": source_duration,
        "output_sample_rate": config.target_sample_rate, "output_frames": len(mono),
        "output_duration_seconds": len(mono) / config.target_sample_rate,
        "resampled": source_rate != config.target_sample_rate,
        "resample_ratio": {"up": up, "down": down},
        "pre_crop_frames": before_crop, "crop_start_frame": start,
        "crop_stop_frame": stop, "cropped": before_crop > limit, "padded": False,
        "rms": float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))),
        "absolute_peak": float(np.max(np.abs(mono))),
        "processed_float32_le_sha256": hashlib.sha256(mono.astype("<f4").tobytes()).hexdigest(),
    }
    return mono, details


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: str | Path, config: AudioConfig = AudioConfig()) -> tuple[np.ndarray, dict]:
    """Read a bounded local recording and return waveform plus processing metadata."""
    path = Path(path)
    try:
        with sf.SoundFile(path) as stream:
            if stream.samplerate <= 0 or stream.frames / stream.samplerate > config.max_source_duration_seconds:
                raise ValueError("Source exceeds the duration guard; refusing full decode")
            source = {"format": stream.format, "subtype": stream.subtype}
            frames, rate = stream.frames, stream.samplerate
            samples = stream.read(dtype="float32", always_2d=True)
            if len(samples) != frames:
                raise ValueError("Audio decoder returned incomplete frame data")
    except (sf.SoundFileError, OSError) as exc:
        raise ValueError(f"Could not decode audio file {path.name!r}: {exc}") from exc
    waveform, details = process_samples(samples, rate, config)
    report = {"filename": path.name, "file_sha256": _file_hash(path),
              "config": asdict(config), "source_encoding": source,
              "policies": {"mono": "channel_mean", "resampling": "polyphase_kaiser_5",
                           "resampling_boundary": "constant_zero", "duration": "center_crop_no_padding",
                           "silence": "retain", "amplitude": "preserve_no_normalization_or_clipping"},
              **details}
    return waveform, report
