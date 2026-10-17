"""Exact recording-content audit; never train, tune, or score evaluation audio."""

import csv
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import AudioConfig, _file_hash, process_samples
from .feature_cache import _vector_hash, runtime_identity
from .manifest import REQUIRED_COLUMNS, validate_manifest
from .protocols import _write_csv, parse_protocol


HASH_FIELDS = ("encoded_sha256", "decoded_sha256", "processed_sha256")


def waveform_digest(samples: np.ndarray, sample_rate: int, dtype: str) -> str:
    """Domain-separated hash includes rate, shape and canonical little-endian sample bytes."""
    values = np.asarray(samples)
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all()
            or type(sample_rate) is not int or sample_rate <= 0 or dtype not in ("<f4", "<f8")):
        raise ValueError("Invalid waveform fingerprint input")
    canonical = np.array(values, dtype=dtype, order="C", copy=True)
    canonical[canonical == 0] = 0  # Positive and negative floating zero represent the same silence.
    header = {"schema": "waveform-v1", "sample_rate": sample_rate,
              "frames": len(canonical), "channels": canonical.shape[1], "dtype": dtype}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def fingerprint_recording(path: str | Path, config: AudioConfig = AudioConfig()) -> dict:
    path = Path(path)
    encoded = _file_hash(path)
    try:
        with sf.SoundFile(path) as stream:
            rate, frames, channels = stream.samplerate, stream.frames, stream.channels
            if rate <= 0 or channels <= 0 or not config.min_duration_seconds <= frames / rate <= config.max_source_duration_seconds:
                raise ValueError("Source duration is outside configured limits; refusing full decode")
            samples = stream.read(dtype="float64", always_2d=True)
            encoding = {"format": stream.format, "subtype": stream.subtype}
    except (sf.SoundFileError, OSError) as exc:
        raise ValueError("Could not decode recording for the content audit") from exc
    if samples.shape != (frames, channels):
        raise ValueError("Incomplete decoded frame/channel data")
    waveform, processing = process_samples(samples, rate, config)
    if _file_hash(path) != encoded:
        raise ValueError("Audio source changed during fingerprinting")
    return {"encoded_sha256": encoded, "decoded_sha256": waveform_digest(samples, rate, "<f8"),
            "processed_sha256": waveform_digest(waveform, config.target_sample_rate, "<f4"),
            "encoded_bytes": path.stat().st_size, "source_encoding": encoding,
            "source_sample_rate": rate, "source_frames": frames, "source_channels": channels,
            "output_frames": len(waveform), "resampled": processing["resampled"],
            "cropped": processing["cropped"],
            "processed_float32_le_sha256": processing["processed_float32_le_sha256"]}


def duplicate_groups(records: list[dict], field: str):
    buckets = defaultdict(list)
    for record in records:
        buckets[record[field]].append({name: record[name] for name in REQUIRED_COLUMNS})
    groups = [{"sha256": digest, "members": sorted(members, key=lambda item: item["audio_path"])}
              for digest, members in sorted(buckets.items()) if len(members) > 1]
    cross = sum(len({row["split"] for row in group["members"]}) > 1 for group in groups)
    conflicts = sum(len({row["label"] for row in group["members"]}) > 1 for group in groups)
    summary = {"groups": len(groups), "affected_recordings": sum(len(g["members"]) for g in groups),
               "redundant_rows": sum(len(g["members"]) - 1 for g in groups),
               "cross_split_groups": cross, "within_split_groups": len(groups) - cross,
               "label_conflict_groups": conflicts}
    return summary, groups


def _official_membership(rows, protocols, reference_path):
    if set(protocols) != {"train", "dev", "eval"}:
        raise ValueError("Supply all three official protocol keys for metadata auditing")
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    complete, sources = [], {}
    for split in ("train", "dev", "eval"):
        body = Path(protocols[split]).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != reference["sources"][split]["sha256"]:
            raise ValueError(f"{split} protocol differs from the pinned source hash")
        parsed = parse_protocol(body.decode("utf-8-sig"), split)
        complete.extend(parsed)
        sources[split] = {"sha256": digest, "rows": len(parsed)}
    with tempfile.TemporaryDirectory() as folder:
        combined = Path(folder) / "official.csv"
        _write_csv(combined, complete)
        full_audit = validate_manifest(combined)  # All source speakers checked, before selecting audio.
    official = {row["audio_path"]: row for row in complete}
    for row in rows:
        if official.get(row["audio_path"]) != row:
            raise ValueError("Selected metadata differs from the pinned official recording membership")
    return {"sources": sources, "full_source_metadata_audit": full_audit,
            "official_rows_matched": len(rows), "full_source_speaker_separation_verified": True}


def _feature_evidence(index_path, cache_dir, manifest, rows, config):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if index["manifest_sha256"] != _file_hash(Path(manifest)):
        raise ValueError("Feature index manifest checksum differs")
    if [{name: row[name] for name in REQUIRED_COLUMNS} for row in index["records"]] != rows:
        raise ValueError("Feature index recording membership differs")
    if index["audio_config"] != asdict(config) or index["runtime"] != runtime_identity():
        raise ValueError("Audio recipe or runtime differs from the feature extraction identity")
    root, evidence = Path(cache_dir).resolve(strict=True), {}
    code = {name: _file_hash(Path(__file__).with_name(name))
            for name in ("audio.py", "features.py", "feature_cache.py")}
    for row in index["records"]:
        key = row["cache_key"]
        if not re.fullmatch(r"[0-9a-f]{64}", key) or row["feature_file"] != key + "/features.npy":
            raise ValueError("Invalid feature-cache path")
        entry = (root / key).resolve(strict=True)
        if not entry.is_relative_to(root):
            raise ValueError("Feature entry escapes the cache root")
        metadata = json.loads((entry / "entry.json").read_text(encoding="utf-8"))
        identity = metadata["identity"]
        calculated = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if calculated != key or identity["source_code_sha256"] != code:
            raise ValueError("Feature cache identity or source code differs")
        for name in ("audio_config", "feature_config", "runtime", "extractor_version"):
            if identity[name] != index[name]:
                raise ValueError("Feature cache processing identity differs from index")
        feature_file = entry / "features.npy"
        if not feature_file.resolve(strict=True).is_relative_to(root):
            raise ValueError("Feature file escapes the cache root")
        if _file_hash(feature_file) != metadata["file_sha256"]:
            raise ValueError("Feature file checksum differs")
        vector = np.load(feature_file, allow_pickle=False)
        if vector.dtype != np.float32 or vector.shape != (len(index["feature_names"]),) or not np.isfinite(vector).all():
            raise ValueError("Invalid cached feature vector")
        if _vector_hash(vector) != metadata["vector_sha256"] or metadata["vector_sha256"] != row["vector_sha256"]:
            raise ValueError("Feature vector differs from index")
        evidence[row["audio_path"]] = metadata
    return evidence, code


def audit_recordings(manifest: str | Path, audio_root: str | Path, protocols: dict,
                     protocol_reference: str | Path, index_path: str | Path, cache_dir: str | Path,
                     output_dir: str | Path, config: AudioConfig = AudioConfig()) -> dict:
    target = Path(output_dir)
    if target.exists():
        raise ValueError("Audit output directory already exists; choose a new run directory")
    audit = validate_manifest(manifest)
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        rows = [{name: row[name].strip() for name in REQUIRED_COLUMNS} for row in csv.DictReader(stream)]
    if {row["split"] for row in rows} != {"train", "dev"}:
        raise ValueError("Recording audit requires train/dev only; evaluation audio remains unavailable before freeze")
    membership = _official_membership(rows, protocols, protocol_reference)
    evidence, feature_code = _feature_evidence(index_path, cache_dir, manifest, rows, config)
    root = Path(audio_root).resolve(strict=True)
    paths = [(root / row["audio_path"]).resolve(strict=True) for row in rows]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Resolved audio path escapes the dataset root")
    records = []
    for row, path in zip(rows, paths):
        record = fingerprint_recording(path, config)
        cached = evidence[row["audio_path"]]
        if record["encoded_sha256"] != cached["identity"]["source_sha256"]:
            raise ValueError("Recording bytes differ from the feature-extraction source")
        if record["processed_float32_le_sha256"] != cached["audio_processing"]["processed_float32_le_sha256"]:
            raise ValueError("Audited waveform differs from the cached model input")
        records.append({**row, **record})
    summaries, details = {}, {}
    for field in HASH_FIELDS:
        summaries[field], details[field] = duplicate_groups(records, field)
    any_duplicates = any(value["groups"] for value in summaries.values())
    report = {"schema_version": 1, "scope": "ASVspoof 2019 LA selected train/dev recordings; exact content only",
              "source_manifest_sha256": _file_hash(Path(manifest)), "manifest_audit": audit,
              "source_feature_index_sha256": _file_hash(Path(index_path)),
              "protocol_reference_sha256": _file_hash(Path(protocol_reference)), **membership,
              "audio_config": asdict(config), "runtime": runtime_identity(),
              "feature_source_code_sha256": feature_code,
              "audit_source_sha256": {name: _file_hash(Path(__file__).with_name(name))
                                      for name in ("audit.py", "audio.py", "manifest.py", "protocols.py")},
              "fingerprint_convention": "waveform-v1: compact sorted JSON header + newline + C-order little-endian samples; signed zeros canonicalized",
              "native_decode_dtype": "float64; original rate, frame count and channel order retained",
              "processed_dtype": "float32; existing mono/resample/center-crop recipe",
              "duplicate_summary": summaries, "status": "review_required" if any_duplicates else "pass",
              "cross_split_content_detected": any(value["cross_split_groups"] for value in summaries.values()),
              "contradictory_labels_detected": any(value["label_conflict_groups"] for value in summaries.values()),
              "recordings_checked": len(records), "feature_sources_and_waveforms_verified": len(records),
              "encoded_bytes_checked": sum(record["encoded_bytes"] for record in records),
              "resampled_recordings": sum(record["resampled"] for record in records),
              "cropped_recordings": sum(record["cropped"] for record in records),
              "evaluation_audio_processed": False, "models_refitted": False, "thresholds_changed": False,
              "near_duplicate_detection_performed": False, "recordings_removed": 0}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-audit-", dir=target.parent) as folder:
        staged = Path(folder)
        for name, value in (("recordings.json", records), ("duplicate_groups.json", details)):
            (staged / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["local_artifact_sha256"] = {file.name: _file_hash(file) for file in staged.iterdir()}
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(target)
    return report
