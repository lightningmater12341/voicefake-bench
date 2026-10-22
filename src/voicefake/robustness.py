"""Paired development diagnostics under an already frozen experiment; never tune."""

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .audio import AudioConfig, _file_hash, load_audio
from .audit import waveform_digest
from .baselines import classification_metrics
from .comparison import model_scores
from .compression import CompressionConfig, mp3_roundtrip
from .feature_cache import _vector_hash
from .features import FeatureConfig, extract_features
from .freeze import MODELS, _inside, verify_freeze
from .manifest import REQUIRED_COLUMNS, validate_manifest
from .metrics import roc_summary, threshold_predictions
from .perturbations import PerturbationConfig, apply_perturbation

CONDITIONS = ("clean", "noise_20db", "noise_10db", "bandwidth_8khz", "mp3_32kbps")
STATISTICS = ("roc_auc", "eer", "false_positive_rate", "spoof_recall")


def speaker_draws(rows, replicates, seed):
    """Draw S whole speaker blocks S times; duplicate speakers retain multiplicity."""
    speakers = sorted({row["speaker_id"] for row in rows})
    if len(speakers) < 2 or any(not speaker.strip() for speaker in speakers):
        raise ValueError("Require at least two nonempty speaker IDs")
    if type(replicates) is not int or replicates < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid bootstrap replicate count or seed")
    blocks = [np.array([i for i, row in enumerate(rows) if row["speaker_id"] == speaker]) for speaker in speakers]
    rng = np.random.Generator(np.random.PCG64(seed))
    for draw in rng.integers(0, len(speakers), size=(replicates, len(speakers))):
        yield np.concatenate([blocks[i] for i in draw])


def _statistics(labels, scores, rule):
    ranking = roc_summary(labels, scores)
    operating = classification_metrics(labels, threshold_predictions(scores, rule))
    return {"roc_auc": ranking["roc_auc"], "eer": ranking["eer"],
            "false_positive_rate": operating["false_positive_rate"], "spoof_recall": operating["spoof_recall"]}


def summarize_paired(rows, scores, model_definitions, plan, partition="dev"):
    """All intervals and degraded-minus-clean differences use the same speaker draws."""
    if partition not in ("dev", "eval") or not rows or any(row["split"] != partition for row in rows):
        raise ValueError("Paired diagnostics require only the explicitly declared dev/eval partition")
    if len({row["audio_path"] for row in rows}) != len(rows):
        raise ValueError("Repeated source recording in paired input")
    if any(row["label"] not in ("bonafide", "spoof") for row in rows):
        raise ValueError("Invalid source labels")
    labels = np.array([row["label"] == "spoof" for row in rows], dtype=np.int8)
    if set(labels) != {0, 1}:
        raise ValueError("Require both classes in development input")
    if set(scores) != set(model_definitions) or not scores:
        raise ValueError("Model scores and fixed decision rules differ")
    if plan["threshold_reselection"] is not False or plan["method"] != "paired speaker-block bootstrap":
        raise ValueError("Require the frozen paired bootstrap without threshold reselection")
    if plan["interval"] != "percentile; numpy quantile method=linear" or plan["confidence"] != 0.95:
        raise ValueError("Unsupported interval convention")
    arrays, points, predictions = {}, {}, {}
    for model, conditions in scores.items():
        if set(conditions) != set(CONDITIONS):
            raise ValueError("Missing or extra condition; all sources must stay paired")
        arrays[model], points[model], predictions[model] = {}, {}, {}
        for condition in CONDITIONS:
            array = np.asarray(conditions[condition])
            if array.shape != labels.shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError("Scores must be finite and aligned to every source recording")
            arrays[model][condition] = array.astype(np.float64)
            rule = model_definitions[model]["decision_rule"]
            points[model][condition] = _statistics(labels, array, rule)
            predictions[model][condition] = threshold_predictions(array, rule)
    draws = list(speaker_draws(rows, plan["replicates"], plan["seed"]))
    valid = [indices for indices in draws if set(labels[indices]) == {0, 1}]
    alpha = (1 - plan["confidence"]) / 2
    result = {}
    for model, conditions in arrays.items():
        samples = {condition: {key: [] for key in STATISTICS} for condition in CONDITIONS}
        for indices in valid:
            for condition in CONDITIONS:
                values = _statistics(labels[indices], conditions[condition][indices], model_definitions[model]["decision_rule"])
                for key in STATISTICS:
                    samples[condition][key].append(values[key])
        result[model] = {}
        for condition in CONDITIONS:
            stats = {}
            for key in STATISTICS:
                values = np.asarray(samples[condition][key])
                difference = values - np.asarray(samples["clean"][key])
                interval = lambda data: (np.quantile(data, [alpha, 1-alpha], method="linear").tolist() if len(data) else None)
                stats[key] = {"estimate": points[model][condition][key], "interval_95": interval(values),
                              "difference_from_clean": points[model][condition][key] - points[model]["clean"][key],
                              "paired_difference_interval_95": interval(difference)}
            predicted = predictions[model][condition]
            clean = predictions[model]["clean"]
            attacks = {}
            for attack in sorted({r["attack_id"] for r in rows if r["label"] == "spoof"}):
                mask = np.array([r["label"] == "spoof" and r["attack_id"] == attack for r in rows])
                detected = int(predicted[mask].sum())
                attacks[attack] = {"spoof_count": int(mask.sum()), "detected": detected, "spoof_recall": detected / int(mask.sum())}
            result[model][condition] = {"statistics": stats,
                "confusion_matrix": classification_metrics(labels, predicted)["confusion_matrix"],
                "per_attack": attacks,
                "paired_decision_changes": {"genuine_new_false_alarms": int(np.sum((labels == 0) & (clean == 0) & (predicted == 1))),
                                            "genuine_false_alarms_removed": int(np.sum((labels == 0) & (clean == 1) & (predicted == 0))),
                                            "spoofs_newly_detected": int(np.sum((labels == 1) & (clean == 0) & (predicted == 1))),
                                            "spoofs_no_longer_detected": int(np.sum((labels == 1) & (clean == 1) & (predicted == 0)))}}
    draw_digest = hashlib.sha256()
    for indices in draws:
        draw_digest.update(len(indices).to_bytes(8, "little")); draw_digest.update(indices.astype("<i8").tobytes())
    return {"models": result, "bootstrap": {**plan, "speakers": len({r["speaker_id"] for r in rows}),
            "requested_draws": len(draws), "valid_draws": len(valid), "invalid_draw_policy": plan["invalid_draws"], "invalid_draws": len(draws)-len(valid),
            "draw_indices_sha256": draw_digest.hexdigest(), "rng": "numpy.Generator(PCG64)",
            "speaker_order": "sorted IDs; uniform sampling with replacement; S blocks per draw",
            "estimand": "clip-weighted metrics after speaker-block resampling; all conditions and models share draws",
            "per_attack_intervals_computed": False}}


def run_robustness(repo_root, output_dir="artifacts/robustness/day11"):
    root = Path(repo_root).resolve(strict=True)
    target = _inside(root, output_dir)
    if not target.is_relative_to(root / "artifacts") or target.exists():
        raise ValueError("Choose a new output directory inside artifacts")
    spec_path = root / "configs/frozen-experiment.json"
    verified = verify_freeze(spec_path, root)  # Before any audio decoding or model deserialization.
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    spec = read("configs/frozen-experiment.json")
    if tuple(spec["conditions"]) != CONDITIONS or tuple(spec["compared_models"]) != MODELS:
        raise ValueError("Unsupported frozen model/condition set")
    manifest_name = "data/manifests/baseline-train-dev.csv"
    validate_manifest(root / manifest_name)
    with (root / manifest_name).open(newline="", encoding="utf-8-sig") as file:
        all_rows = [{key: row[key].strip() for key in REQUIRED_COLUMNS} for row in csv.DictReader(file)]
    if {row["split"] for row in all_rows} != {"train", "dev"}:
        raise ValueError("Require the frozen train/dev manifest; evaluation audio is forbidden")
    rows = [row for row in all_rows if row["split"] == "dev"]
    if len(rows) != 700:
        raise ValueError("Expected all 700 original development recordings")
    evidence_files = (("artifacts/audits/day8/recordings.json", "docs/RECORDING_AUDIT_RESULTS.json", "recordings.json"),
                      ("artifacts/perturbations/day9/transforms.json", "docs/PERTURBATION_AUDIT.json", "transforms.json"),
                      ("artifacts/compression/day10/recordings.json", "docs/COMPRESSION_AUDIT.json", "recordings.json"))
    evidence, evidence_hashes = [], {}
    for path, report, key in evidence_files:
        digest = _file_hash(root / path)
        if digest != read(report)["local_artifact_sha256"][key]:
            raise ValueError("Prior recording/transform evidence changed")
        records = read(path)
        if [{key: r[key] for key in REQUIRED_COLUMNS} for r in records] != all_rows:
            raise ValueError("Prior recording/transform membership differs")
        evidence.append({r["audio_path"]: r for r in records})
        evidence_hashes[path] = digest
    index = read("artifacts/features/baseline-index.json")
    indexed = {r["audio_path"]: r for r in index["records"]}
    old_score_path = root / "artifacts/comparisons/day6-final/development_predictions.npz"
    old_metrics = read("docs/METRIC_RESULTS.json")
    if _file_hash(old_score_path) != old_metrics["source_development_scores_sha256"]:
        raise ValueError("Prior clean development predictions changed")
    with np.load(old_score_path, allow_pickle=False) as old:
        old_scores = {name: old[name + "_scores"].copy() for name in MODELS}
        old_labels = old["labels"].copy()
    labels = np.array([r["label"] == "spoof" for r in rows], dtype=np.int8)
    if not np.array_equal(old_labels, labels):
        raise ValueError("Prior clean prediction label order differs")
    audio_config = AudioConfig.from_json(root / "configs/audio.json")
    feature_config = FeatureConfig.from_json(root / "configs/features.json")
    perturb_config = PerturbationConfig.from_json(root / "configs/perturbations.json")
    codec_config = CompressionConfig.from_json(root / "configs/compression.json")
    audio_root = (root / "data/asvspoof2019/LA").resolve(strict=True)
    vectors = {condition: [] for condition in CONDITIONS}
    details = []
    with threadpool_limits(limits=1):
        for row in rows:
            path = (audio_root / row["audio_path"]).resolve(strict=True)
            if not path.is_relative_to(audio_root):
                raise ValueError("Recording escapes the dataset root")
            clean, loaded = load_audio(path, audio_config)
            prior = evidence[0][row["audio_path"]]
            clean_digest = waveform_digest(clean, 16000, "<f4")
            if loaded["file_sha256"] != prior["encoded_sha256"] or clean_digest != prior["processed_sha256"]:
                raise ValueError("Clean waveform differs from audited model input")
            record = {**row, "conditions": {}}
            for condition in CONDITIONS:
                if condition == "clean":
                    waveform, transform = clean, {"output_sha256": clean_digest}
                elif condition == "mp3_32kbps":
                    waveform, transform = mp3_roundtrip(clean, codec_config)
                    expected = {k: v for k, v in evidence[2][row["audio_path"]].items() if k not in REQUIRED_COLUMNS}
                    if transform != expected:
                        raise ValueError("MP3 replay differs from pre-freeze audit")
                else:
                    waveform, transform = apply_perturbation(clean, condition, perturb_config)
                    if transform != evidence[1][row["audio_path"]]["conditions"][condition]:
                        raise ValueError("Perturbation replay differs from pre-freeze audit")
                vector, _ = extract_features(waveform, feature_config.sample_rate, feature_config)
                digest = _vector_hash(vector)
                if condition == "clean" and digest != indexed[row["audio_path"]]["vector_sha256"]:
                    raise ValueError("Clean feature replay differs from original model inputs")
                vectors[condition].append(vector)
                record["conditions"][condition] = {"waveform_sha256": transform["output_sha256"], "feature_sha256": digest}
            if waveform_digest(clean, 16000, "<f4") != clean_digest:
                raise ValueError("A transform changed its clean source")
            details.append(record)
        scores = {}
        for name in MODELS:
            model = joblib.load(_inside(root, spec["models"][name]["artifact"]))
            kind = "margin" if name == "svm" else "probability"
            scores[name] = {}
            for condition in CONDITIONS:
                features = np.stack(vectors[condition]).astype(np.float64)
                first = model_scores(model, features, kind)
                repeated = model_scores(model, features, kind)
                if not np.array_equal(first, repeated):
                    raise ValueError("Prediction replay differed")
                if name != "svm" and (np.any(first < 0) or np.any(first > 1)):
                    raise ValueError("Invalid probability scores")
                scores[name][condition] = first
            if not np.array_equal(scores[name]["clean"], old_scores[name]):
                raise ValueError("Clean scores differ from the pre-freeze comparison")
    summary = summarize_paired(rows, scores, spec["models"], spec["uncertainty_plan"])
    for name in MODELS:
        actual = summary["models"][name]["clean"]
        prior = old_metrics["models"][name]
        if actual["confusion_matrix"] != prior["operating_point"]["metrics"]["confusion_matrix"]:
            raise ValueError("Clean operating metrics differ from original threshold evidence")
        if actual["statistics"]["roc_auc"]["estimate"] != prior["roc_auc"] or actual["statistics"]["eer"]["estimate"] != prior["eer"]:
            raise ValueError("Clean ranking metrics differ from original evidence")
    verify_freeze(spec_path, root)
    report = {"schema_version": 1, "scope": "paired ASVspoof 2019 LA development subset; known attacks only; diagnostic after selection",
              "recordings": len(rows), "genuine": int(np.sum(labels == 0)), "spoof": int(np.sum(labels == 1)),
              "conditions": list(CONDITIONS), "primary_model": spec["primary_model"],
              "frozen_lock_sha256": verified["lock_sha256"], "frozen_file_bindings_verified": verified["file_bindings_verified"],
              "source_evidence_sha256": evidence_hashes, "source_clean_scores_sha256": _file_hash(old_score_path),
              "source_code_sha256": {"robustness.py": _file_hash(Path(__file__))},
              "audio_runtime": spec["audio_runtime"], "model_runtime": spec["model_runtime"], "codec": spec["codec"],
              "decision_rules": {name: spec["models"][name]["decision_rule"] for name in MODELS},
              "models_refitted": False, "thresholds_changed": False, "new_model_selection_performed": False,
              "evaluation_audio_processed": False, "waveform_files_retained": False,
              "clean_features_and_scores_match_pre_freeze": True, "all_transform_fingerprints_match_pre_freeze": True,
              "identical_prediction_replays": len(MODELS)*len(CONDITIONS), **summary}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-robustness-", dir=target.parent) as folder:
        staged = Path(folder)
        (staged / "recordings.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
        np.savez_compressed(staged / "scores.npz", labels=labels,
                            **{model+"__"+condition: array for model, conditions in scores.items() for condition, array in conditions.items()})
        report["local_artifact_sha256"] = {p.name: _file_hash(p) for p in staged.iterdir()}
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
        staged.rename(target)
    return report
