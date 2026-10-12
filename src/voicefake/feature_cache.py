"""Content-keyed feature caches and auditable manifest indexes."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import AudioConfig, _file_hash, load_audio
from .features import EXTRACTOR_VERSION, FeatureConfig, extract_features, feature_names
from .manifest import REQUIRED_COLUMNS, validate_manifest


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "libsndfile": sf.__libsndfile_version__,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "soundfile")}}


def _vector_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest()


def cached_features(audio_path: str | Path, cache_dir: str | Path,
                    audio_config: AudioConfig = AudioConfig(),
                    feature_config: FeatureConfig = FeatureConfig()) -> tuple[np.ndarray, dict]:
    """Reuse only a matching, hash-verified entry; corrupt entries fail explicitly."""
    if audio_config.target_sample_rate != feature_config.sample_rate:
        raise ValueError("Audio target rate and feature sample rate must match")
    path = Path(audio_path)
    identity = {"extractor_version": EXTRACTOR_VERSION, "source_sha256": _file_hash(path),
                "audio_config": asdict(audio_config), "feature_config": asdict(feature_config),
                "runtime": runtime_identity(),
                "source_code_sha256": {name: _file_hash(Path(__file__).with_name(name))
                                       for name in ("audio.py", "features.py", "feature_cache.py")}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    root = Path(cache_dir)
    entry = root / key

    def read_entry():
        try:
            metadata = json.loads((entry / "entry.json").read_text(encoding="utf-8"))
            if metadata["identity"] != identity or metadata["file_sha256"] != _file_hash(entry / "features.npy"):
                raise ValueError("Cache identity or file checksum mismatch")
            vector = np.load(entry / "features.npy", allow_pickle=False)
            if vector.dtype != np.float32 or vector.shape != (2 * feature_config.n_mfcc,) or not np.isfinite(vector).all():
                raise ValueError("Cached feature shape, type, or values are invalid")
            if metadata["vector_sha256"] != _vector_hash(vector):
                raise ValueError("Cached vector checksum mismatch")
            return vector, {"cache_key": key, "cache_hit": True, **metadata}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid cache entry {key}: {exc}") from exc

    if entry.exists():
        return read_entry()
    waveform, audio_report = load_audio(path, audio_config)
    if audio_report["file_sha256"] != identity["source_sha256"]:
        raise ValueError("Audio source changed while extracting features")
    vector, details = extract_features(waveform, audio_config.target_sample_rate, feature_config)
    audio_report.pop("filename")
    metadata = {"identity": identity, "vector_sha256": _vector_hash(vector),
                "details": details, "audio_processing": audio_report}
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-", dir=root) as folder:
        staged = Path(folder)
        np.save(staged / "features.npy", vector, allow_pickle=False)
        metadata["file_sha256"] = _file_hash(staged / "features.npy")
        (staged / "entry.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        try:
            staged.rename(entry)
        except FileExistsError:
            return read_entry()
    return vector, {"cache_key": key, "cache_hit": False, **metadata}


def extract_manifest(manifest: str | Path, audio_root: str | Path, cache_dir: str | Path,
                     output: str | Path, audio_config: AudioConfig = AudioConfig(),
                     feature_config: FeatureConfig = FeatureConfig(), allow_eval: bool = False) -> dict:
    """Write an index with labels separate from numeric features; never fit a scaler."""
    output = Path(output)
    if output.exists():
        raise ValueError("Feature index already exists; choose a new output path")
    audit = validate_manifest(manifest)
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        rows = [{name: row[name].strip() for name in REQUIRED_COLUMNS} for row in csv.DictReader(stream)]
    if not allow_eval and any(row["split"] == "eval" for row in rows):
        raise ValueError("Evaluation feature extraction requires --allow-eval after experiment freeze")
    root = Path(audio_root).resolve(strict=True)
    records = []
    for row in rows:
        path = (root / row["audio_path"].replace("\\", "/")).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Resolved audio path escapes the dataset root")
        vector, cached = cached_features(path, cache_dir, audio_config, feature_config)
        records.append({**row, "cache_key": cached["cache_key"],
                        "feature_file": cached["cache_key"] + "/features.npy",
                        "vector_sha256": cached["vector_sha256"], "cache_hit": cached["cache_hit"],
                        "frame_count": cached["details"]["frame_count"]})
    report = {"schema_version": 1, "extractor_version": EXTRACTOR_VERSION,
              "manifest_sha256": _file_hash(Path(manifest)), "manifest_audit": audit,
              "audio_config": asdict(audio_config), "feature_config": asdict(feature_config),
              "runtime": runtime_identity(), "feature_names": feature_names(feature_config),
              "feature_shape": [len(records), 2 * feature_config.n_mfcc],
              "cache_hits": sum(record["cache_hit"] for record in records),
              "records": records, "learned_preprocessing_fitted": False,
              "models_trained": False, "evaluation_audio_processed": any(r["split"] == "eval" for r in records)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report
