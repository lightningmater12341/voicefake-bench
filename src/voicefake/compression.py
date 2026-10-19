"""Fixed, gapless MP3 round trip with explicit reversible codec-domain headroom."""

import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import AudioConfig, _file_hash, load_audio
from .audit import waveform_digest
from .feature_cache import runtime_identity
from .manifest import REQUIRED_COLUMNS, validate_manifest


@dataclass(frozen=True)
class CompressionConfig:
    schema_version: int = 1
    sample_rate: int = 16000
    bitrate_kbps: int = 32
    algorithm_quality: int = 2
    headroom_gain: float = 0.5

    def __post_init__(self):
        expected = {"schema_version": 1, "sample_rate": 16000, "bitrate_kbps": 32, "algorithm_quality": 2}
        if any(type(getattr(self, key)) is not int or getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("Version 1 fixes mono 16 kHz, 32 kbps CBR, algorithm quality 2")
        if type(self.headroom_gain) not in (int, float) or self.headroom_gain != 0.5:
            raise ValueError("Version 1 fixes headroom gain 0.5 and inverse gain 2.0")

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Compression configuration must be an object")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown compression configuration fields") from exc


def _ffmpeg_path():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(arguments, payload=None):
    try:
        result = subprocess.run([_ffmpeg_path(), *arguments], input=payload, capture_output=True,
                                timeout=30, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("FFmpeg unavailable or timed out; compression is not verified") from exc
    if result.returncode:
        # No local paths from codec diagnostics enter public evidence.
        raise ValueError("FFmpeg codec command failed; no compression result accepted")
    return result.stdout


@lru_cache(maxsize=1)
def codec_identity() -> dict:
    version = _run(["-version"]).decode("utf-8").splitlines()[0]
    encoder = _run(["-hide_banner", "-h", "encoder=libmp3lame"]).decode("utf-8")
    if "libmp3lame" not in encoder or "fltp" not in encoder:
        raise ValueError("FFmpeg build must provide the floating-point libmp3lame encoder")
    return {"package": "imageio-ffmpeg", "package_version": importlib.metadata.version("imageio-ffmpeg"),
            "ffmpeg_version": version, "binary_sha256": _file_hash(Path(_ffmpeg_path())),
            "encoder": "libmp3lame", "decoder": "mp3float", "encoder_help_sha256": hashlib.sha256(encoder.encode()).hexdigest()}


def mp3_roundtrip(waveform: np.ndarray, config: CompressionConfig = CompressionConfig()):
    clean = np.asarray(waveform)
    if clean.ndim != 1 or clean.dtype != np.float32 or not 4000 <= len(clean) <= 160000 or not np.isfinite(clean).all() or not np.any(clean):
        raise ValueError("Require finite nonzero mono float32 audio, 0.25-10 seconds at 16 kHz")
    scaled = (clean * config.headroom_gain).astype("<f4")
    if float(np.max(np.abs(scaled))) >= 1:
        raise ValueError("Fixed headroom exhausted; refuse encoding rather than clip or adapt gain")
    codec_identity()
    # A declared frame-aligned zero extension avoids the observed short-final-frame gapless discrepancy.
    # Verify the decoder against this known codec input length before removing only our own extension.
    tail_padding = (-len(clean)) % 576  # MPEG-2 Layer III at 16 kHz: 576 samples per frame.
    codec_input = np.pad(scaled, (0, tail_padding), mode="constant")
    with tempfile.TemporaryDirectory(prefix="voicefake-codec-") as folder:
        path = Path(folder) / "audio.mp3"  # Seekable output retains Xing/LAME delay/padding metadata.
        _run(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
              "-map", "0:a:0", "-c:a", "libmp3lame", "-b:a", "32k", "-compression_level", "2", "-reservoir", "1", "-abr", "0",
              "-ar", "16000", "-ac", "1", "-threads", "1", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
              "-write_xing", "1", "-id3v2_version", "0", "-write_id3v1", "0", "-f", "mp3", str(path)], codec_input.tobytes())
        encoded = path.read_bytes()
        decoded = _run(["-hide_banner", "-loglevel", "error", "-nostdin", "-c:a", "mp3float", "-i", str(path),
                        "-map", "0:a:0", "-ar", "16000", "-ac", "1", "-threads", "1", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
    if len(decoded) != len(codec_input) * 4:
        raise ValueError("Gapless decoder changed the declared codec-input length; refusing any alignment repair")
    output = (np.frombuffer(decoded, dtype="<f4")[:len(clean)] / config.headroom_gain).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite codec output")
    return output, {"condition": "mp3_32kbps", "clean_sha256": waveform_digest(clean, 16000, "<f4"),
                    "encoded_sha256": hashlib.sha256(encoded).hexdigest(), "encoded_bytes": len(encoded),
                    "output_sha256": waveform_digest(output, 16000, "<f4"), "output_frames": len(output),
                    "sample_rate": 16000, "headroom_input_peak": float(np.max(np.abs(scaled))),
                    "absolute_output_peak": float(np.max(np.abs(output))),
                    "above_unit_output_samples": int(np.sum(np.abs(output) > 1)),
                    "protocol_zero_tail_samples": tail_padding, "gapless_codec_frames": len(codec_input),
                    "alignment_search_performed": False, "clipping_applied": False,
                    "headroom_gain": 0.5, "inverse_gain": 2.0}


def audit_compression(manifest, audio_root, recording_audit, recordings, output_dir,
                      config: CompressionConfig = CompressionConfig()):
    target = Path(output_dir)
    if target.exists():
        raise ValueError("Compression output already exists")
    metadata = validate_manifest(manifest)
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        rows = [{key: row[key].strip() for key in REQUIRED_COLUMNS} for row in csv.DictReader(stream)]
    if {row["split"] for row in rows} != {"train", "dev"}:
        raise ValueError("Compression audit requires train/dev only; evaluation audio is forbidden before freeze")
    source = json.loads(Path(recording_audit).read_text(encoding="utf-8"))
    if source["status"] != "pass" or source["source_manifest_sha256"] != _file_hash(Path(manifest)) or source["manifest_audit"] != metadata:
        raise ValueError("Prior recording audit does not match a clean source manifest")
    if _file_hash(Path(recordings)) != source["local_artifact_sha256"]["recordings.json"]:
        raise ValueError("Recording evidence checksum differs")
    evidence = json.loads(Path(recordings).read_text(encoding="utf-8"))
    if [{key: row[key] for key in REQUIRED_COLUMNS} for row in evidence] != rows:
        raise ValueError("Recording evidence membership differs")
    if source["runtime"] != runtime_identity() or source["audit_source_sha256"]["audio.py"] != _file_hash(Path(__file__).with_name("audio.py")):
        raise ValueError("Clean loader/runtime differs")
    root = Path(audio_root).resolve(strict=True)
    paths = [(root / row["audio_path"]).resolve(strict=True) for row in rows]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Resolved recording escapes the audio root")
    records = []
    for row, prior, path in zip(rows, evidence, paths):
        clean, loaded = load_audio(path, AudioConfig(**source["audio_config"]))
        identity = waveform_digest(clean, 16000, "<f4")
        if loaded["file_sha256"] != prior["encoded_sha256"] or identity != prior["processed_sha256"]:
            raise ValueError("Clean audio no longer matches audited model input")
        output, details = mp3_roundtrip(clean, config)
        replay, repeated = mp3_roundtrip(clean, config)
        if details != repeated or not np.array_equal(output, replay):
            raise ValueError("Codec bitstream or decoded replay differed")
        if waveform_digest(clean, 16000, "<f4") != identity:
            raise ValueError("Codec modified its source")
        records.append({**row, **details})
    report = {"schema_version": 1, "scope": "MP3 property audit on the existing ASVspoof 2019 LA train/dev subset",
              "config": asdict(config), "codec": codec_identity(), "runtime": runtime_identity(),
              "manifest_audit": metadata, "recordings_checked": len(rows), "identical_bitstream_and_sample_replays": len(rows),
              "maximum_headroom_input_peak": max(row["headroom_input_peak"] for row in records),
              "maximum_decoded_output_peak": max(row["absolute_output_peak"] for row in records),
              "above_unit_output_samples": sum(row["above_unit_output_samples"] for row in records),
              "gapless_sample_counts_verified": len(rows),
              "protocol_tail_padding_samples": sum(row["protocol_zero_tail_samples"] for row in records), "source_manifest_sha256": _file_hash(Path(manifest)),
              "source_recording_audit_sha256": _file_hash(Path(recording_audit)),
              "source_recordings_sha256": _file_hash(Path(recordings)),
              "source_code_sha256": {name: _file_hash(Path(__file__).with_name(name)) for name in ("compression.py", "audio.py", "audit.py")},
              "evaluation_audio_processed": False, "detector_scores_computed": False, "models_refitted": False,
              "thresholds_changed": False, "waveform_or_mp3_files_retained": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-compression-", dir=target.parent) as folder:
        staged = Path(folder)
        (staged / "recordings.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["local_artifact_sha256"] = {"recordings.json": _file_hash(staged / "recordings.json")}
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(target)
    return report
