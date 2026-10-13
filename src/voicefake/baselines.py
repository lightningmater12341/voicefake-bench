"""Train/development baselines on verified feature indexes; evaluation stays unused."""

import csv
import hashlib
import importlib.metadata
import json
import math
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .audio import _file_hash
from .manifest import REQUIRED_COLUMNS, validate_manifest


@dataclass(frozen=True)
class BaselineConfig:
    schema_version: int = 1
    C: float = 1.0
    max_iter: int = 2000
    tolerance: float = 1e-4
    class_weight: str | None = "balanced"
    seed: int = 42
    min_class_count: int = 20

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported baseline configuration schema")
        for value in (self.C, self.tolerance):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Regularization C and tolerance must be positive finite numbers")
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(self.min_class_count) is not int or self.min_class_count < 2:
            raise ValueError("min_class_count must be at least two")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be balanced or null")

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Baseline configuration must be a JSON object")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown baseline configuration fields") from exc


def classification_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict:
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if labels.ndim != 1 or labels.shape != predicted.shape or set(labels) != {0, 1} or not set(predicted).issubset({0, 1}):
        raise ValueError("Metrics require both binary classes and aligned binary predictions")
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    genuine_recall, spoof_recall = tn / (tn + fp), tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"samples": len(labels), "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": (tn + tp) / len(labels),
            "balanced_accuracy": (genuine_recall + spoof_recall) / 2,
            "genuine_recall": genuine_recall, "spoof_recall": spoof_recall,
            "false_positive_rate": fp / (tn + fp), "spoof_precision": precision, "spoof_f1": f1}


def fit_baselines(x_train: np.ndarray, y_train: np.ndarray, x_dev: np.ndarray,
                  y_dev: np.ndarray, config: BaselineConfig = BaselineConfig()):
    arrays = [np.asarray(x, dtype=np.float64) for x in (x_train, x_dev)]
    labels = [np.asarray(y) for y in (y_train, y_dev)]
    for name, x, y in zip(("train", "dev"), arrays, labels):
        if x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError(f"{name}: invalid feature/label arrays")
        if set(y) != {0, 1} or min(int(np.sum(y == label)) for label in (0, 1)) < config.min_class_count:
            raise ValueError(f"{name}: require at least {config.min_class_count} examples of each class")
    if arrays[0].shape[1] != arrays[1].shape[1]:
        raise ValueError("Train and development feature counts differ")
    dummy = DummyClassifier(strategy="most_frequent")
    logistic = Pipeline([
        ("float64", FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}, validate=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=config.C, solver="lbfgs", max_iter=config.max_iter,
                                          tol=config.tolerance, class_weight=config.class_weight,
                                          random_state=config.seed)),
    ])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            dummy.fit(arrays[0], labels[0])
            logistic.fit(arrays[0], labels[0])
        except ConvergenceWarning as exc:
            raise ValueError("Logistic regression did not converge; no results will be published") from exc
        models = {"majority": dummy, "logistic": logistic}
        metrics, scores = {}, {}
        for name, model in models.items():
            scores[name] = {}
            metrics[name] = {}
            position = list(model.classes_).index(1)
            for split, x, y in zip(("train", "dev"), arrays, labels):
                score = model.predict_proba(x)[:, position]
                if not np.isfinite(score).all():
                    raise ValueError("Model produced non-finite probabilities")
                scores[name][split] = score
                metrics[name][split] = classification_metrics(y, (score >= 0.5).astype(np.int8))
    scaler = logistic.named_steps["scaler"]
    if not np.allclose(scaler.mean_, arrays[0].mean(axis=0), rtol=0, atol=1e-12):
        raise ValueError("Scaler statistics differ from training data")
    details = {"metrics": metrics, "decision_rule": "spoof probability >= 0.5; untuned",
               "logistic_iterations": logistic.named_steps["classifier"].n_iter_.tolist(),
               "scaler_fit_rows": int(scaler.n_samples_seen_), "scaler_fit_split": "train",
               "scaler_training_statistics_verified": True}
    return models, scores, details


def load_feature_partitions(index_path: str | Path, manifest: str | Path, cache_dir: str | Path):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    audit = validate_manifest(manifest)
    if index["manifest_sha256"] != _file_hash(Path(manifest)):
        raise ValueError("Feature index and source manifest hashes differ")
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        source_rows = [{name: row[name].strip() for name in REQUIRED_COLUMNS} for row in csv.DictReader(stream)]
    indexed_rows = [{name: row[name] for name in REQUIRED_COLUMNS} for row in index["records"]]
    if source_rows != indexed_rows:
        raise ValueError("Index labels or recording membership differ from source manifest")
    if set(row["split"] for row in source_rows) != {"train", "dev"}:
        raise ValueError("Baseline input must contain only train and dev, both present")
    if index["feature_shape"] != [len(source_rows), len(index["feature_names"])]:
        raise ValueError("Feature-index dimensions are inconsistent")
    root = Path(cache_dir).resolve(strict=True)
    vectors = {"train": [], "dev": []}
    labels = {"train": [], "dev": []}
    source_splits = {}
    feature_code = None
    for row in index["records"]:
        key = row["cache_key"]
        if not re.fullmatch(r"[0-9a-f]{64}", key) or row["feature_file"] != key + "/features.npy":
            raise ValueError("Invalid feature-cache path in index")
        entry = root / key
        if not entry.resolve(strict=True).is_relative_to(root):
            raise ValueError("Cache entry resolves outside the cache root")
        metadata = json.loads((entry / "entry.json").read_text(encoding="utf-8"))
        identity = metadata["identity"]
        if feature_code is None:
            feature_code = identity["source_code_sha256"]
        elif feature_code != identity["source_code_sha256"]:
            raise ValueError("Feature entries were computed by different source code")
        calculated_key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if calculated_key != key:
            raise ValueError("Cache identity does not match its key")
        for field in ("audio_config", "feature_config", "runtime", "extractor_version"):
            if identity[field] != index[field]:
                raise ValueError("Cache and index processing identities differ")
        path = entry / "features.npy"
        if metadata["file_sha256"] != _file_hash(path):
            raise ValueError("Cached feature file checksum mismatch")
        vector = np.load(path, allow_pickle=False)
        digest = hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest()
        if vector.dtype != np.float32 or vector.shape != (len(index["feature_names"]),) or not np.isfinite(vector).all():
            raise ValueError("Cached feature shape, dtype, or values are invalid")
        if digest != metadata["vector_sha256"] or digest != row["vector_sha256"]:
            raise ValueError("Cached vector checksum differs from index")
        split = row["split"]
        previous = source_splits.setdefault(identity["source_sha256"], split)
        if previous != split:
            raise ValueError("Identical encoded audio appears across train/dev")
        vectors[split].append(vector)
        labels[split].append(1 if row["label"] == "spoof" else 0)
    x = {split: np.stack(value) for split, value in vectors.items()}
    y = {split: np.asarray(value, dtype=np.int8) for split, value in labels.items()}
    index["verified_feature_code_sha256"] = feature_code
    return x, y, index, audit


def run_baselines(index_path: str | Path, manifest: str | Path, cache_dir: str | Path,
                  output_dir: str | Path, config: BaselineConfig = BaselineConfig()) -> dict:
    target = Path(output_dir)
    if target.exists():
        raise ValueError("Baseline output directory already exists; choose a new run directory")
    x, y, index, audit = load_feature_partitions(index_path, manifest, cache_dir)
    models, scores, details = fit_baselines(x["train"], y["train"], x["dev"], y["dev"], config)
    report = {"schema_version": 1, "scope": "ASVspoof 2019 LA train/development subset; known attacks only",
              "positive_label": "spoof=1; bonafide=0", "config": asdict(config),
              "model_definitions": {"majority": "most-frequent training class",
                                    "logistic": "float64 cast + training-fitted StandardScaler + L2 logistic regression, lbfgs"},
              "source_manifest_sha256": _file_hash(Path(manifest)),
              "feature_index_sha256": _file_hash(Path(index_path)), "manifest_audit": audit,
              "feature_names": index["feature_names"], "feature_config": index["feature_config"],
              "feature_source_code_sha256": index["verified_feature_code_sha256"],
              "model_input_dtype": "float64",
              "audio_config": index["audio_config"], "feature_runtime": index["runtime"],
              "model_runtime": {name: importlib.metadata.version(name)
                                for name in ("scikit-learn", "joblib", "numpy", "scipy", "threadpoolctl")},
              "trainer_source_sha256": _file_hash(Path(__file__)),
              **details, "development_used_for_fit": False, "evaluation_audio_processed": False,
              "hyperparameter_search_performed": False, "operating_threshold_selected": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-baseline-", dir=target.parent) as folder:
        staged = Path(folder)
        for name, model in models.items():
            joblib.dump(model, staged / f"{name}.joblib")
        np.savez_compressed(staged / "development_predictions.npz", labels=y["dev"],
                            majority_scores=scores["majority"]["dev"], logistic_scores=scores["logistic"]["dev"])
        report["local_artifact_sha256"] = {file.name: _file_hash(file) for file in staged.iterdir()}
        (staged / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.rename(target)
    return report
