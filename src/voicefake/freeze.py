"""Seal dataset/model/threshold/condition choices before any evaluation audio is scored."""

import csv
import hashlib
import importlib.metadata
import io
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .audio import _file_hash
from .compression import codec_identity
from .feature_cache import runtime_identity
from .manifest import REQUIRED_COLUMNS, validate_manifest
from .protocols import _write_csv, parse_protocol, select_subset


CRITICAL_CODE = ("audio.py", "features.py", "feature_cache.py", "manifest.py", "protocols.py",
                 "baselines.py", "comparison.py", "metrics.py", "audit.py", "perturbations.py", "compression.py", "freeze.py")
CONFIGS = ("audio", "features", "baseline", "comparison", "metrics", "perturbations", "compression")
EVIDENCE = ("PROTOCOL_AUDIT", "COMPARISON_RESULTS", "METRIC_RESULTS", "RECORDING_AUDIT_RESULTS", "PERTURBATION_AUDIT", "COMPRESSION_AUDIT")
MODELS = ("majority", "logistic", "svm", "random_forest")


def _inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative:
        raise ValueError("Frozen bindings must be repository-relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Frozen binding escapes the repository")
    return resolved


def seal_specification(body: dict) -> dict:
    if "lock_sha256" in body:
        raise ValueError("Cannot reseal an existing lock")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "lock_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def verify_freeze(specification: str | Path, repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve(strict=True)
    spec = json.loads(Path(specification).read_text(encoding="utf-8"))
    digest = spec.pop("lock_sha256", None)
    if digest != seal_specification(spec)["lock_sha256"]:
        raise ValueError("Frozen specification content checksum differs")
    if spec.get("schema_version") != 1 or spec.get("status") != "frozen":
        raise ValueError("Expected a frozen version-1 specification")
    checked = 0
    for group in ("code_config_evidence", "local_artifacts"):
        if not spec[group]:
            raise ValueError("Frozen file bindings must not be empty")
        for name, expected in spec[group].items():
            if _file_hash(_inside(root, name)) != expected:
                raise ValueError(f"Frozen binding changed: {name}")
            checked += 1
    if spec["codec"] != codec_identity() or spec["audio_runtime"] != runtime_identity():
        raise ValueError("Frozen codec/audio runtime differs")
    current = {name: importlib.metadata.version(name) for name in spec["model_runtime"]}
    if current != spec["model_runtime"]:
        raise ValueError("Frozen model runtime differs")
    return {"status": "verified", "lock_sha256": digest, "file_bindings_verified": checked,
            "primary_model": spec["primary_model"], "evaluation_audio_processed": False}


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return [{key: row[key].strip() for key in REQUIRED_COLUMNS} for row in csv.DictReader(stream)]


def create_freeze(repo_root: str | Path, output: str = "configs/frozen-experiment.json",
                  evaluation_manifest: str = "data/manifests/frozen-eval.csv") -> dict:
    root = Path(repo_root).resolve(strict=True)
    target, evaluation = _inside(root, output), _inside(root, evaluation_manifest)
    if target.exists() or evaluation.exists():
        raise ValueError("Frozen specification or evaluation manifest already exists; never overwrite a freeze")
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    protocol = read("docs/PROTOCOL_AUDIT.json")
    comparison = read("docs/COMPARISON_RESULTS.json")
    metrics = read("docs/METRIC_RESULTS.json")
    compression = read("docs/COMPRESSION_AUDIT.json")
    if compression["evaluation_audio_processed"] is not False or compression["detector_scores_computed"] is not False:
        raise ValueError("Compression validation must precede evaluation and detector scoring")
    if compression["codec"] != codec_identity() or compression["config"] != read("configs/compression.json"):
        raise ValueError("Compression audit recipe/codec differs")
    if compression["recordings_checked"] != 1400 or compression["identical_bitstream_and_sample_replays"] != 1400:
        raise ValueError("Expected completed compression validation on all 1,400 train/dev clips")
    for name, digest in compression["source_code_sha256"].items():
        if _file_hash(root / "src/voicefake" / name) != digest:
            raise ValueError("Compression source changed after its audit")
    for report, field in ((comparison, "trainer_source_sha256"), (metrics, "metric_source_sha256")):
        for name, digest in report[field].items():
            if _file_hash(root / "src/voicefake" / name) != digest:
                raise ValueError("Model/metric implementation changed after its development evidence")
    perturbation = read("docs/PERTURBATION_AUDIT.json")
    if perturbation["config"] != read("configs/perturbations.json"):
        raise ValueError("Perturbation configuration changed after validation")
    for name, digest in perturbation["source_code_sha256"].items():
        if _file_hash(root / "src/voicefake" / name) != digest:
            raise ValueError("Perturbation implementation changed after validation")
    if comparison["development_leader"] != "svm" or metrics["prior_development_leader"] != "svm":
        raise ValueError("Primary model must match the original development selection")
    if compression["runtime"] != runtime_identity():
        raise ValueError("Compression runtime changed after validation")
    if {name: importlib.metadata.version(name) for name in metrics["source_model_runtime"]} != metrics["source_model_runtime"]:
        raise ValueError("Model package runtime changed before freezing")
    complete, protocol_files = [], {}
    for split, info in protocol["sources"].items():
        name = "data/asvspoof2019/" + info["filename"]
        path = _inside(root, name)
        if _file_hash(path) != info["sha256"]:
            raise ValueError("Pinned protocol key changed")
        complete.extend(parse_protocol(path.read_text(encoding="utf-8-sig"), split))
        protocol_files[name] = info["sha256"]
    with tempfile.TemporaryDirectory() as directory:
        full = Path(directory) / "source.csv"; _write_csv(full, complete)
        if validate_manifest(full) != protocol["full_input_audit"]:
            raise ValueError("Full official metadata audit differs")
    selected = select_subset(complete, 100, 42)
    subset_name = "data/manifests/smoke.csv"
    train_name = "data/manifests/baseline-train-dev.csv"
    if _file_hash(root / subset_name) != protocol["manifest_sha256"] or _read_rows(root / subset_name) != selected:
        raise ValueError("Deterministic subset differs from the original declared selection")
    train_rows = [row for row in selected if row["split"] in ("train", "dev")]
    if _read_rows(root / train_name) != train_rows:
        raise ValueError("Train/dev membership differs from the declared subset")
    train_hash = _file_hash(root / train_name)
    if any(report["source_manifest_sha256"] != train_hash for report in (comparison, metrics, compression)):
        raise ValueError("Model, threshold and codec evidence refer to different training/development samples")
    eval_rows = [row for row in selected if row["split"] == "eval"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
    writer.writeheader(); writer.writerows(eval_rows)
    eval_bytes = stream.getvalue().encode("utf-8")
    local = {name: _file_hash(root / name) for name in (train_name, subset_name, "artifacts/features/baseline-index.json")}
    local.update(protocol_files)
    model_details = {}
    for name in MODELS:
        path = "artifacts/comparisons/day6-final/" + name + ".joblib"
        digest = _file_hash(root / path)
        if digest != metrics["source_model_sha256"][name]:
            raise ValueError("Model artifact changed since development selection")
        local[path] = digest
        point = metrics["models"][name]
        model_details[name] = {"artifact": path, "sha256": digest, "score_kind": point["score_kind"],
                               "decision_rule": point["operating_point"]["decision_rule"],
                               "selection_target_fpr": point["operating_point"]["target_fpr"]}
    if local["artifacts/features/baseline-index.json"] != metrics["source_feature_index_sha256"]:
        raise ValueError("Clean feature index changed since development scoring")
    if len(eval_rows) != 1400 or sum(row["label"] == "bonafide" for row in eval_rows) != 100:
        raise ValueError("Expected the originally declared 100-genuine/1,300-spoof evaluation subset")
    eval_name = evaluation.relative_to(root).as_posix()
    local[eval_name] = hashlib.sha256(eval_bytes).hexdigest()
    bindings = {"src/voicefake/" + name: _file_hash(root / "src/voicefake" / name) for name in CRITICAL_CODE}
    bindings.update({"configs/" + name + ".json": _file_hash(root / "configs" / (name + ".json")) for name in CONFIGS})
    bindings.update({"docs/" + name + ".json": _file_hash(root / "docs" / (name + ".json")) for name in EVIDENCE})
    body = {"schema_version": 1, "status": "frozen", "experiment_id": "voicefake-la-subset-v1",
            "frozen_date_Asia_Kolkata": datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat(),
            "positive_label": "spoof=1; bonafide=0", "primary_model": "svm", "compared_models": list(MODELS),
            "primary_selection_rule": "Day 6 highest development balanced accuracy at default thresholds; exact ties keep declared order",
            "selected_model_parameters": comparison["selected"], "model_config": metrics["source_model_config"],
            "models": model_details, "threshold_policy": "Day 7 genuine-only clean-development threshold; carry unchanged to every condition and evaluation; threshold ties are spoof",
            "dataset": {"doi": protocol["dataset_doi"], "license": protocol["dataset_license"],
                        "selection": protocol["selection"], "train": {"bonafide": 100, "spoof": 600},
                        "dev": {"bonafide": 100, "spoof": 600}, "eval": {"bonafide": 100, "spoof": 1300},
                        "evaluation_manifest": eval_name, "official_speakers_disjoint": True,
                        "evaluation_content_audit_required_before_scoring": True},
            "conditions": ["clean", "noise_20db", "noise_10db", "bandwidth_8khz", "mp3_32kbps"],
            "condition_composition": "each condition separately starts from the clean processed waveform; no training augmentation",
            "compression_policy": {"headroom_gain": 0.5, "inverse_gain": 2.0, "codec_frame_samples": 576,
                                   "tail_rule": "zero-extend codec input to a multiple of 576; verify gapless decoded padded length; remove only the known extension",
                                   "out_of_range_policy": "abort if scaled input peak >= 1; no adaptive gain or clipping; float output remains unclipped"},
            "metric_policy": {"roc_auc": "tie-grouped trapezoidal ROC", "eer": "linear adjacent-ROC FPR=FNR crossing; no deployment threshold inferred",
                              "operating_metrics": ["false_positive_rate", "spoof_recall", "confusion_matrix"],
                              "per_attack_recall": "report source attack IDs with counts; no method-specific tuning"},
            "uncertainty_plan": {"method": "paired speaker-block bootstrap", "replicates": 1000, "seed": 42,
                                 "confidence": 0.95, "interval": "percentile; numpy quantile method=linear",
                                 "threshold_reselection": False, "invalid_draws": "exclude draws missing either class; report invalid/valid counts without replacing draws"},
            "evaluation_policy": "one frozen reporting run after content audit; no tuning or silent exclusions; report failures and actual realized FPR",
            "codec": compression["codec"], "audio_runtime": runtime_identity(), "model_runtime": metrics["source_model_runtime"],
            "code_config_evidence": bindings, "local_artifacts": local,
            "evaluation_audio_processed_at_freeze": False}
    spec = seal_specification(body)
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with evaluation.open("xb") as file:
        file.write(eval_bytes)
    with target.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    verify_freeze(target, root)
    return spec
