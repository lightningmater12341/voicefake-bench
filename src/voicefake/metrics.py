"""Tie-aware binary ranking metrics and development-only operating points."""

import csv
import importlib.metadata
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .audio import _file_hash
from .baselines import classification_metrics
from .manifest import validate_manifest


@dataclass(frozen=True)
class MetricConfig:
    schema_version: int = 1
    target_fpr: float = 0.01

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported metric configuration schema")
        if type(self.target_fpr) not in (int, float) or not math.isfinite(self.target_fpr) or not 0 <= self.target_fpr < 1:
            raise ValueError("target_fpr must be finite and in [0, 1)")

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Metric configuration must be a JSON object")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown metric configuration fields") from exc


def _scores(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iuf":
        raise ValueError("Scores must be a nonempty one-dimensional real numeric array")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Scores must be finite")
    return array


def binary_roc(labels, scores):
    """Return FPR, TPR, thresholds with complete score-tie blocks and both endpoints."""
    scores = _scores(scores)
    labels = np.asarray(labels)
    if labels.shape != scores.shape or labels.dtype.kind not in "biuf" or set(labels) != {0, 1}:
        raise ValueError("Require aligned binary labels with both bonafide=0 and spoof=1")
    order = np.argsort(scores, kind="stable")[::-1]
    ordered = scores[order]
    # Inspect only the last member of each score block: never break a tied score by row order.
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(ordered) - 1]
    tp = np.r_[0, np.cumsum(labels[order] == 1, dtype=np.int64)[ends]]
    fp = np.r_[0, np.cumsum(labels[order] == 0, dtype=np.int64)[ends]]
    return fp / int(np.sum(labels == 0)), tp / int(np.sum(labels == 1)), np.r_[np.inf, ordered[ends]]


def roc_summary(labels, scores) -> dict:
    fpr, tpr, thresholds = binary_roc(labels, scores)
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    fnr = 1 - tpr
    difference = fpr - fnr
    upper = int(np.flatnonzero(difference >= 0)[0])
    lower = upper if difference[upper] == 0 else upper - 1
    weight = (0.0 if lower == upper else float(-difference[lower] / (difference[upper] - difference[lower])))
    eer = float(fpr[lower] + weight * (fpr[upper] - fpr[lower]))
    return {"roc_auc": auc, "eer": eer, "roc_points": len(thresholds),
            "eer_convention": "linear interpolation of adjacent tie-grouped ROC points where FPR=FNR; no operating threshold inferred",
            "eer_bracket": {"lower_fpr": float(fpr[lower]), "lower_fnr": float(fnr[lower]),
                            "upper_fpr": float(fpr[upper]), "upper_fnr": float(fnr[upper]),
                            "interpolation_weight": weight}}


def threshold_predictions(scores, decision_rule: dict) -> np.ndarray:
    """Apply a saved rule unchanged to new finite scores; this function selects nothing."""
    scores = _scores(scores)
    if decision_rule.get("mode") == "reject_all" and decision_rule.get("threshold") is None:
        return np.zeros(len(scores), dtype=np.int8)
    threshold = decision_rule.get("threshold")
    if (decision_rule.get("mode") != "score_at_least" or type(threshold) not in (int, float)
            or not math.isfinite(threshold)):
        raise ValueError("Invalid saved decision rule")
    return (scores >= threshold).astype(np.int8)


def select_development_threshold(genuine_scores, config: MetricConfig = MetricConfig()) -> dict:
    """Select from genuine development scores alone; spoof scores cannot affect the cutoff."""
    genuine = np.sort(_scores(genuine_scores))[::-1]
    fraction = Fraction(str(config.target_fpr))
    budget = len(genuine) * fraction.numerator // fraction.denominator
    cutoff = float(genuine[budget])
    # Smallest float64 above the first excluded genuine score. All its ties stay excluded.
    with np.errstate(over="ignore"):
        threshold = float(np.nextafter(cutoff, np.inf))
    rule = {"mode": "score_at_least", "threshold": threshold} if math.isfinite(threshold) else {
        "mode": "reject_all", "threshold": None}
    false_positives = int(threshold_predictions(genuine, rule).sum())
    if false_positives > budget:
        raise ValueError("Selected threshold exceeded the integer false-positive budget")
    return {"target_fpr": config.target_fpr, "genuine_selection_count": len(genuine),
            "false_positive_budget": budget, "genuine_cutoff_score": cutoff,
            "decision_rule": rule, "selection_split": "dev", "selection_labels_used": ["bonafide"],
            "tie_policy": "exclude the entire boundary genuine-score block; score equal to saved threshold is spoof",
            "method": "budget=floor(decimal target_fpr * genuine_count); next float64 above descending genuine[budget]"}


def score_development(run_dir: str | Path, manifest: str | Path, output_dir: str | Path,
                      config: MetricConfig = MetricConfig()) -> dict:
    target, source = Path(output_dir), Path(run_dir)
    if target.exists():
        raise ValueError("Metric output directory already exists; choose a new run directory")
    prior = json.loads((source / "report.json").read_text(encoding="utf-8"))
    audit = validate_manifest(manifest)
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if {row["split"].strip() for row in rows} != {"train", "dev"}:
        raise ValueError("Threshold selection requires train/dev provenance; evaluation membership is forbidden")
    if prior["source_manifest_sha256"] != _file_hash(Path(manifest)) or prior["manifest_audit"] != audit:
        raise ValueError("Manifest differs from the source comparison run")
    if prior["evaluation_audio_processed"] is not False or prior["development_used_for_fit"] is not False:
        raise ValueError("Source comparison has incompatible data boundaries")
    if prior["positive_label"] != "spoof=1; bonafide=0":
        raise ValueError("Source comparison has incompatible positive-label orientation")
    models = ("majority", "logistic", "svm", "random_forest")
    if set(prior["score_rules"]) != set(models) or set(prior["metrics"]) != set(models):
        raise ValueError("Expected the four documented comparison models")
    prediction_path = source / "development_predictions.npz"
    if _file_hash(prediction_path) != prior["local_artifact_sha256"][prediction_path.name]:
        raise ValueError("Development score artifact checksum mismatch")
    expected = np.array([1 if row["label"].strip() == "spoof" else 0
                         for row in rows if row["split"].strip() == "dev"], dtype=np.int8)
    results = {}
    with np.load(prediction_path, allow_pickle=False) as saved:
        if set(saved.files) != {"labels", *(name + "_scores" for name in models)}:
            raise ValueError("Development score artifact has unexpected arrays")
        if saved["labels"].shape != expected.shape or not np.array_equal(saved["labels"], expected):
            raise ValueError("Saved development labels do not match manifest order")
        for name in models:
            scores = _scores(saved[name + "_scores"])
            ranking = roc_summary(expected, scores)  # Also validates aligned labels and both classes.
            old_rule = prior["score_rules"][name]
            kind = "svm_decision_margin" if name == "svm" else "spoof_probability"
            if old_rule["score_kind"] != kind or old_rule["tie"] != "spoof":
                raise ValueError("Source score orientation or tie convention differs")
            if kind == "spoof_probability" and (np.any(scores < 0) or np.any(scores > 1)):
                raise ValueError("Probability score outside [0, 1]")
            old_metrics = classification_metrics(expected, threshold_predictions(scores, {
                "mode": "score_at_least", "threshold": old_rule["threshold"]}))
            if old_metrics != prior["metrics"][name]["dev"]:
                raise ValueError("Saved scores do not reproduce source development metrics")
            point = select_development_threshold(scores[expected == 0], config)
            point["metrics"] = classification_metrics(expected, threshold_predictions(scores, point["decision_rule"]))
            results[name] = {"score_kind": kind, **ranking, "default_threshold_metrics": old_metrics,
                             "operating_point": point}
    report = {"schema_version": 1, "scope": "ASVspoof 2019 LA selected development subset; known attacks only",
              "positive_label": "spoof=1; bonafide=0", "config": asdict(config), "models": results,
              "source_manifest_sha256": _file_hash(Path(manifest)), "manifest_audit": audit,
              "source_comparison_report_sha256": _file_hash(source / "report.json"),
              "source_development_scores_sha256": _file_hash(prediction_path),
              "source_feature_index_sha256": prior["feature_index_sha256"],
              "source_model_sha256": {name: prior["local_artifact_sha256"][name + ".joblib"] for name in models},
              "source_model_config": prior["config"], "source_model_runtime": prior["model_runtime"],
              "metric_runtime": {"numpy": importlib.metadata.version("numpy")},
              "metric_source_sha256": {name: _file_hash(Path(__file__).with_name(name))
                                       for name in ("metrics.py", "baselines.py")},
              "models_refitted": False, "new_model_selection_performed": False,
              "prior_model_selection_metric": "development balanced accuracy at default thresholds",
              "prior_development_leader": prior["development_leader"],
              "thresholds_selected_on_development": True, "evaluation_audio_processed": False,
              "experiment_frozen": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-metrics-", dir=target.parent) as folder:
        staged = Path(folder)
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(target)
    return report
