"""One frozen evaluation report after official-membership and recording-content audits."""

import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .audio import AudioConfig, _file_hash, load_audio
from .audit import HASH_FIELDS, _official_membership, duplicate_groups, fingerprint_recording, waveform_digest
from .comparison import model_scores
from .compression import CompressionConfig, mp3_roundtrip
from .feature_cache import _vector_hash, runtime_identity
from .features import FeatureConfig, extract_features
from .freeze import MODELS, _inside, verify_freeze
from .manifest import REQUIRED_COLUMNS, validate_manifest
from .perturbations import PerturbationConfig, apply_perturbation
from .robustness import CONDITIONS, summarize_paired


def _read(root, name):
    return json.loads(_inside(root, name).read_text(encoding="utf-8"))


def _rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return [{key: row[key].strip() for key in REQUIRED_COLUMNS} for row in csv.DictReader(file)]


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _save_arrays(path, arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _evaluation_rows(root, spec):
    path = _inside(root, spec["dataset"]["evaluation_manifest"])
    validate_manifest(path)
    rows = _rows(path)
    if (len(rows) != 1400 or {r["split"] for r in rows} != {"eval"}
            or sum(r["label"] == "bonafide" for r in rows) != 100
            or {r["attack_id"] for r in rows if r["label"] == "spoof"} != {f"A{i:02d}" for i in range(7,20)}
            or any(sum(r["attack_id"] == f"A{i:02d}" and r["label"] == "spoof" for r in rows) != 100 for i in range(7,20))):
        raise ValueError("Require exactly the original frozen 100-genuine/1,300-spoof evaluation selection")
    return path, rows


def audit_evaluation(repo_root, output_dir="artifacts/evaluation-audit/day13", progress=None):
    root = Path(repo_root).resolve(strict=True)
    target = _inside(root, output_dir)
    if not target.is_relative_to(root / "artifacts") or target.exists():
        raise ValueError("Choose a new evaluation-audit directory inside artifacts")
    verified = verify_freeze(root / "configs/frozen-experiment.json", root)
    spec = _read(root, "configs/frozen-experiment.json")
    manifest, rows = _evaluation_rows(root, spec)
    protocol = _read(root, "docs/PROTOCOL_AUDIT.json")
    keys = {split: _inside(root, "data/asvspoof2019/" + info["filename"]) for split, info in protocol["sources"].items()}
    official = _official_membership(rows, keys, root / "docs/PROTOCOL_AUDIT.json")
    baseline_path = root / "data/manifests/baseline-train-dev.csv"
    baseline = _rows(baseline_path)
    # Also validate the selected union, not merely each partition on its own.
    with tempfile.TemporaryDirectory() as folder:
        union = Path(folder) / "selected.csv"
        with union.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
            writer.writeheader(); writer.writerows(baseline + rows)
        selected_audit = validate_manifest(union)
    prior_path = root / "artifacts/audits/day8/recordings.json"
    prior_report = _read(root, "docs/RECORDING_AUDIT_RESULTS.json")
    if _file_hash(prior_path) != prior_report["local_artifact_sha256"]["recordings.json"]:
        raise ValueError("Prior train/dev content evidence changed")
    previous = _read(root, "artifacts/audits/day8/recordings.json")
    if [{k:r[k] for k in REQUIRED_COLUMNS} for r in previous] != baseline:
        raise ValueError("Prior train/dev content membership differs")
    receipt_name = "data/asvspoof2019/frozen-eval-acquisition.json"
    receipt = _read(root, receipt_name)
    public_receipt = _read(root, "docs/EVALUATION_ACQUISITION.json")
    if (_file_hash(root / receipt_name) != public_receipt["local_acquisition_report_sha256"]
            or receipt["source_manifest_sha256"] != _file_hash(manifest)
            or receipt["frozen_lock_sha256"] != verified["lock_sha256"] or receipt["all_crc_checks_passed"] is not True):
        raise ValueError("Acquisition receipt differs from frozen membership or verified archive evidence")
    if [r["audio_path"] for r in receipt["checks"]] != [r["audio_path"] for r in rows]:
        raise ValueError("Acquisition receipt has missing, reordered or extra selected recordings")
    acquired = {r["audio_path"]: r for r in receipt["checks"]}
    audio_root = (root / "data/asvspoof2019/LA").resolve(strict=True)
    for record in previous:
        path = (audio_root / record["audio_path"]).resolve(strict=True)
        if not path.is_relative_to(audio_root) or _file_hash(path) != record["encoded_sha256"]:
            raise ValueError("Current train/dev recording no longer matches the trained/audited source")
    config = AudioConfig.from_json(root / "configs/audio.json")
    records, failures = [], []
    for index, row in enumerate(rows):
        try:
            path = (audio_root / row["audio_path"]).resolve(strict=True)
            if not path.is_relative_to(audio_root):
                raise ValueError("Recording escapes the audio root")
            result = fingerprint_recording(path, config)
            if result["encoded_sha256"] != acquired[row["audio_path"]]["sha256"] or result["encoded_bytes"] != acquired[row["audio_path"]]["bytes"]:
                raise ValueError("Recording differs from archive-checked acquisition")
            records.append({**row, **result})
        except (OSError, ValueError) as exc:
            failures.append({**row, "error_type": type(exc).__name__, "stage": "recording-content audit"})
        if progress is not None and (index+1) % 200 == 0:
            progress(f"Evaluation content audited {index+1}/{len(rows)}")
    duplicate_summary, details = {}, {}
    for field in HASH_FIELDS:
        duplicate_summary[field], details[field] = duplicate_groups(previous + records, field)
    requires_review = failures or any(s["groups"] for s in duplicate_summary.values())
    report = {"schema_version": 1, "scope": "frozen evaluation recording-content audit against selected train/dev; no detector scores",
              "status": "review_required" if requires_review else "pass", "frozen_lock_sha256": verified["lock_sha256"],
              "source_eval_manifest_sha256": _file_hash(manifest), "selected_evaluation_recordings": len(rows),
              "evaluation_recordings_fingerprinted": len(records), "baseline_recordings_reverified": len(previous),
              "recordings_checked_in_union": len(previous)+len(records), "failures": len(failures),
              "selected_union_manifest_audit": selected_audit, **official, "duplicate_summary": duplicate_summary,
              "prior_recordings_sha256": _file_hash(prior_path), "acquisition_report_sha256": _file_hash(root / receipt_name),
              "source_code_sha256": {"evaluation.py": _file_hash(Path(__file__))}, "runtime": runtime_identity(),
              "evaluation_audio_processed": True, "detector_scores_computed": False, "models_refitted": False,
              "thresholds_changed": False, "recordings_removed": 0, "near_duplicate_detection_performed": False,
              "encoded_eval_bytes": sum(r["encoded_bytes"] for r in records),
              "resampled_recordings": sum(r["resampled"] for r in records), "cropped_recordings": sum(r["cropped"] for r in records)}
    verify_freeze(root / "configs/frozen-experiment.json", root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-eval-audit-", dir=target.parent) as folder:
        staged = Path(folder)
        for name, value in (("recordings.json", records), ("duplicate_groups.json", details), ("failures.json", failures)):
            _write_json(staged/name, value)
        report["local_artifact_sha256"] = {p.name: _file_hash(p) for p in staged.iterdir()}
        _write_json(staged/"report.json", report)
        staged.rename(target)
    return report


def _claim_run(root, target, identity, resume):
    """One claim per frozen experiment, even if another output directory is requested."""
    claims = root / "artifacts/evaluation/claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim = claims / (identity["frozen_lock_sha256"] + ".json")
    if claim.exists():
        prior = json.loads(claim.read_text(encoding="utf-8"))
        if prior["identity"] != identity or prior["output_dir"] != target.relative_to(root).as_posix():
            raise ValueError("Frozen evaluation already claimed; cannot start another run or change its inputs/code")
        if prior["status"] == "complete":
            raise ValueError("Frozen evaluation already complete; use its saved scores/report")
        if not resume:
            raise ValueError("Incomplete frozen run exists; explicitly resume that same run")
        if not target.is_dir():
            raise ValueError("Claimed evaluation output is missing; preserve the existing claim for review")
        return claim, prior
    if resume or target.exists():
        raise ValueError("Cannot resume an unclaimed run or overwrite an existing output")
    value = {"schema_version": 1, "status": "started", "identity": identity,
             "output_dir": target.relative_to(root).as_posix(), "started_utc": datetime.now(timezone.utc).isoformat(), "score_files": {}}
    # Exclusive creation prevents two processes from claiming this experiment concurrently.
    with claim.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, indent=2)+"\n")
    target.mkdir(parents=True)
    return claim, value


def run_evaluation(repo_root, output_dir="artifacts/evaluation/frozen-v1", audit_dir="artifacts/evaluation-audit/day13", resume=False, progress=None):
    root = Path(repo_root).resolve(strict=True)
    target = _inside(root, output_dir)
    if not target.is_relative_to(root / "artifacts"):
        raise ValueError("Evaluation output must be inside artifacts")
    verified = verify_freeze(root / "configs/frozen-experiment.json", root)
    spec = _read(root, "configs/frozen-experiment.json")
    if tuple(spec["conditions"]) != CONDITIONS or tuple(spec["compared_models"]) != MODELS:
        raise ValueError("Unexpected frozen conditions/models")
    manifest, rows = _evaluation_rows(root, spec)
    audit_path = _inside(root, audit_dir)
    audit = _read(root, audit_path.relative_to(root).as_posix()+"/report.json")
    records_path = audit_path / "recordings.json"
    if (audit["status"] != "pass" or audit["failures"] != 0 or audit["evaluation_recordings_fingerprinted"] != len(rows)
            or audit["frozen_lock_sha256"] != verified["lock_sha256"] or audit["source_eval_manifest_sha256"] != _file_hash(manifest)
            or audit["source_code_sha256"]["evaluation.py"] != _file_hash(Path(__file__)) or audit["runtime"] != runtime_identity()
            or _file_hash(records_path) != audit["local_artifact_sha256"]["recordings.json"]):
        raise ValueError("Require the matching passing evaluation content audit before any scores")
    audited = _read(root, records_path.relative_to(root).as_posix())
    if [{k:r[k] for k in REQUIRED_COLUMNS} for r in audited] != rows:
        raise ValueError("Audited evaluation membership/order differs")
    identity = {"frozen_lock_sha256": verified["lock_sha256"], "evaluation_manifest_sha256": _file_hash(manifest),
                "content_audit_sha256": _file_hash(audit_path / "report.json"),
                "source_code_sha256": {name: _file_hash(Path(__file__).with_name(name)) for name in ("evaluation.py", "robustness.py")}}
    claim, journal = _claim_run(root, target, identity, resume)
    def notify(message):
        if progress is not None: progress(message)
    try:
        # Recheck exact encoded sources even when resuming from saved features.
        audio_root = (root / "data/asvspoof2019/LA").resolve(strict=True)
        paths = [(audio_root / row["audio_path"]).resolve(strict=True) for row in rows]
        if any(not p.is_relative_to(audio_root) or _file_hash(p) != old["encoded_sha256"] for p, old in zip(paths, audited)):
            raise ValueError("Evaluation source changed after its content audit")
        feature_path = target / "features.npz"
        detail_path = target / "recordings.json"
        if "feature_files" in journal:
            for name, digest in journal["feature_files"].items():
                if _file_hash(target / name) != digest: raise ValueError("Saved evaluation feature evidence changed")
            with np.load(feature_path, allow_pickle=False) as saved:
                vectors = {c: saved[c].copy() for c in CONDITIONS}
            details = json.loads(detail_path.read_text(encoding="utf-8"))
        else:
            audio = AudioConfig.from_json(root / "configs/audio.json")
            features = FeatureConfig.from_json(root / "configs/features.json")
            perturb = PerturbationConfig.from_json(root / "configs/perturbations.json")
            compression = CompressionConfig.from_json(root / "configs/compression.json")
            vectors = {c: [] for c in CONDITIONS}; details = []; failures = []
            with threadpool_limits(limits=1):
                for index, (row, path, old) in enumerate(zip(rows, paths, audited)):
                    clean, loaded = load_audio(path, audio)
                    digest = waveform_digest(clean, audio.target_sample_rate, "<f4")
                    if loaded["file_sha256"] != old["encoded_sha256"] or digest != old["processed_sha256"]:
                        raise ValueError("Evaluation waveform differs from the audited model input")
                    detail = {**row, "conditions": {}}
                    for condition in CONDITIONS:
                        try:
                            if condition == "clean": wave, change = clean, {"output_sha256": digest}
                            elif condition == "mp3_32kbps": wave, change = mp3_roundtrip(clean, compression)
                            else: wave, change = apply_perturbation(clean, condition, perturb)
                            vector, _ = extract_features(wave, features.sample_rate, features)
                            vectors[condition].append(vector)
                            detail["conditions"][condition] = {"waveform_sha256": change["output_sha256"], "feature_sha256": _vector_hash(vector)}
                        except (OSError, ValueError) as exc:
                            failures.append({**row, "condition": condition, "error_type": type(exc).__name__})
                    if waveform_digest(clean, audio.target_sample_rate, "<f4") != digest:
                        raise ValueError("A condition modified its clean source")
                    details.append(detail)
                    if (index+1) % 100 == 0: notify(f"Evaluation features {index+1}/{len(rows)}; failures {len(failures)}")
            _write_json(target / "processing_failures.json", failures)
            if failures:
                raise ValueError("Evaluation processing failures preserved; no recordings dropped and no model scores computed")
            vectors = {c: np.stack(values) for c, values in vectors.items()}
            _save_arrays(feature_path, vectors); _write_json(detail_path, details)
            journal["feature_files"] = {p.name: _file_hash(p) for p in (feature_path, detail_path)}
            journal["status"] = "features_complete"; _write_json(claim, journal)
        for values in vectors.values():
            if values.shape != (len(rows), 40) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Invalid saved evaluation feature matrix")
        scores = {}
        with threadpool_limits(limits=1):
            for name in MODELS:
                scores[name] = {}
                model = None
                for condition in CONDITIONS:
                    key = name + "__" + condition
                    path = target / (key + ".npz")
                    if key in journal["score_files"]:
                        if _file_hash(path) != journal["score_files"][key]: raise ValueError("Saved evaluation scores changed")
                        with np.load(path, allow_pickle=False) as saved: value = saved["scores"].copy()
                    else:
                        if model is None: model = joblib.load(_inside(root, spec["models"][name]["artifact"]))
                        value = model_scores(model, vectors[condition].astype(np.float64), "margin" if name == "svm" else "probability")
                        _save_arrays(path, {"scores": value})
                        journal["score_files"][key] = _file_hash(path); journal["status"] = "scoring"; _write_json(claim, journal)
                    if value.shape != (len(rows),) or not np.isfinite(value).all() or (name != "svm" and (np.any(value < 0) or np.any(value > 1))):
                        raise ValueError("Invalid evaluation score array")
                    scores[name][condition] = value
                notify(f"Frozen evaluation scored {name} on all five conditions")
        summary = summarize_paired(rows, scores, spec["models"], spec["uncertainty_plan"], partition="eval")
        verified = verify_freeze(root / "configs/frozen-experiment.json", root)
        labels = np.array([r["label"] == "spoof" for r in rows], dtype=np.int8)
        _save_arrays(target / "scores.npz", {"labels": labels, **{m+"__"+c: value for m, conditions in scores.items() for c, value in conditions.items()}})
        report = {"schema_version": 1, "scope": "single frozen ASVspoof 2019 LA evaluation subset; 100 genuine and 100 per attack A07-A19; not full challenge",
                  "status": "complete", "started_utc": journal["started_utc"], "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "frozen_lock_sha256": verified["lock_sha256"], "frozen_file_bindings_verified": verified["file_bindings_verified"],
                  "primary_model": spec["primary_model"], "recordings": len(rows), "genuine": 100, "spoof": 1300,
                  "conditions": list(CONDITIONS), "decision_rules": {m: spec["models"][m]["decision_rule"] for m in MODELS},
                  "identity": identity, "source_content_audit_sha256": _file_hash(audit_path / "report.json"),
                  "audio_runtime": spec["audio_runtime"], "model_runtime": spec["model_runtime"], "codec": spec["codec"],
                  "evaluation_audio_processed": True, "models_refitted": False, "thresholds_changed": False,
                  "evaluation_used_for_selection": False, "recordings_removed": 0, "processing_failures": 0,
                  "waveform_files_retained": False, "bootstrap_thresholds_reselected": False, **summary}
        report["local_artifact_sha256"] = {p.name: _file_hash(p) for p in (target/"scores.npz",feature_path,detail_path,target/"processing_failures.json")}
        _write_json(target / "report.json", report)
        journal.update(status="complete", report_sha256=_file_hash(target/"report.json"), finished_utc=report["finished_utc"])
        journal.pop("last_error_type", None); journal.pop("last_error_message", None); _write_json(claim, journal)
        return report
    except BaseException as exc:
        journal["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        journal["last_error_type"] = type(exc).__name__
        journal["last_error_message"] = str(exc)  # Local journal only; diagnostic paths never enter public reports.
        _write_json(claim, journal)
        raise
