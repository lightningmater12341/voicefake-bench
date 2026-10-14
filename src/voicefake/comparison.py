"""Bounded train-only fitting with explicit development parameter selection."""

import importlib.metadata
import json
import math
import tempfile
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .audio import _file_hash
from .baselines import BaselineConfig, classification_metrics, fit_baselines, load_feature_partitions


@dataclass(frozen=True)
class ComparisonConfig:
    schema_version: int = 1
    baseline: BaselineConfig = BaselineConfig()
    svm_c: tuple = (0.1, 1.0, 10.0)
    svm_gamma: tuple = ("scale", 0.01)
    svm_tolerance: float = 0.001
    svm_max_iter: int = 10000
    forest_trees: int = 200
    forest_depth: tuple = (None, 12)
    forest_min_leaf: tuple = (1, 3)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported comparison configuration schema")
        if not isinstance(self.baseline, BaselineConfig):
            raise ValueError("baseline must be a BaselineConfig")
        for name in ("svm_c", "svm_gamma", "forest_depth", "forest_min_leaf"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
                raise ValueError("Search axes must be nonempty sequences without duplicates")
        for value in self.svm_c:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1000:
                raise ValueError("SVM C must be finite and in (0, 1000]")
        for value in self.svm_gamma:
            if value != "scale" and (type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1000):
                raise ValueError("SVM gamma must be scale or a finite number in (0, 1000]")
        if type(self.svm_tolerance) not in (int, float) or not math.isfinite(self.svm_tolerance) or self.svm_tolerance <= 0:
            raise ValueError("SVM tolerance must be positive and finite")
        if type(self.svm_max_iter) is not int or not 1 <= self.svm_max_iter <= 100000:
            raise ValueError("SVM max_iter must be in [1, 100000]")
        if type(self.forest_trees) is not int or not 1 <= self.forest_trees <= 200:
            raise ValueError("Forest tree count must be in [1, 200]")
        for value in self.forest_depth:
            if value is not None and (type(value) is not int or not 1 <= value <= 32):
                raise ValueError("Forest depth must be null or an integer in [1, 32]")
        for value in self.forest_min_leaf:
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError("Forest minimum leaf size must be in [1, 100]")
        if len(self.svm_c) * len(self.svm_gamma) + len(self.forest_depth) * len(self.forest_min_leaf) > 16:
            raise ValueError("Search is bounded to at most 16 candidates")

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Comparison configuration must be a JSON object")
        try:
            if "baseline" in values:
                values["baseline"] = BaselineConfig(**values["baseline"])
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown or invalid comparison configuration fields") from exc


def model_scores(model, features: np.ndarray, kind: str) -> np.ndarray:
    """Both score types increase toward spoof; margins are not probabilities."""
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected class order bonafide=0, spoof=1")
    values = (model.decision_function(features) if kind == "margin"
              else model.predict_proba(features)[:, 1])
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Model produced invalid scores")
    return values


def select_candidate(candidates: list[dict]) -> int:
    """max keeps the first declared candidate on an exact metric tie."""
    return max(range(len(candidates)), key=lambda i: candidates[i]["metrics"]["dev"]["balanced_accuracy"])


def fit_comparison(x_train, y_train, x_dev, y_dev, config: ComparisonConfig = ComparisonConfig()):
    # This also validates shape, finite values, class labels and minimum class counts.
    models, scores, baseline = fit_baselines(x_train, y_train, x_dev, y_dev, config.baseline)
    x = {"train": np.asarray(x_train, dtype=np.float64), "dev": np.asarray(x_dev, dtype=np.float64)}
    y = {"train": np.asarray(y_train), "dev": np.asarray(y_dev)}
    rules = {name: {"score_kind": "spoof_probability", "threshold": 0.5, "tie": "spoof"}
             for name in models}
    searches, selected, metrics = {}, {}, dict(baseline["metrics"])
    grid = {
        "svm": [{"C": c, "gamma": gamma} for c, gamma in product(config.svm_c, config.svm_gamma)],
        "random_forest": [{"max_depth": depth, "min_samples_leaf": leaf}
                          for depth, leaf in product(config.forest_depth, config.forest_min_leaf)],
    }
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for family, parameters in grid.items():
            candidates, fitted, candidate_scores = [], [], []
            kind, threshold = ("margin", 0.0) if family == "svm" else ("probability", 0.5)
            for params in parameters:
                cast = FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}, validate=True)
                if family == "svm":
                    # Default disables probability fitting across sklearn 1.6-1.9; explicit False is deprecated in 1.9.
                    classifier = SVC(**params, kernel="rbf", class_weight=config.baseline.class_weight,
                                     tol=config.svm_tolerance, max_iter=config.svm_max_iter,
                                     cache_size=128, shrinking=True, random_state=config.baseline.seed)
                    model = Pipeline([("float64", cast), ("scaler", StandardScaler()), ("classifier", classifier)])
                else:
                    classifier = RandomForestClassifier(**params, n_estimators=config.forest_trees,
                                      criterion="gini", max_features="sqrt", bootstrap=True,
                                      class_weight=config.baseline.class_weight, random_state=config.baseline.seed,
                                      n_jobs=1)
                    model = Pipeline([("float64", cast), ("classifier", classifier)])
                try:
                    model.fit(x["train"], y["train"])
                except ConvergenceWarning as exc:
                    raise ValueError("SVM did not converge; no comparison will be published") from exc
                entry = {"parameters": params, "metrics": {}}
                if family == "svm":
                    if classifier.fit_status_ != 0:
                        raise ValueError("SVM fit did not complete")
                    scaler = model.named_steps["scaler"]
                    if int(scaler.n_samples_seen_) != len(x["train"]) or not np.allclose(
                            scaler.mean_, x["train"].mean(axis=0), rtol=0, atol=1e-12):
                        raise ValueError("SVM scaler was not fitted on training statistics")
                    entry.update(scaler_fit_rows=len(x["train"]), scaler_fit_split="train",
                                 scaler_training_statistics_verified=True,
                                 iterations=classifier.n_iter_.tolist())
                values = {split: model_scores(model, features, kind) for split, features in x.items()}
                for split in x:
                    entry["metrics"][split] = classification_metrics(y[split], (values[split] >= threshold).astype(np.int8))
                candidates.append(entry)
                fitted.append(model)
                candidate_scores.append(values)
            winner = select_candidate(candidates)
            searches[family] = candidates
            selected[family] = {"candidate_index": winner, "parameters": parameters[winner]}
            models[family], scores[family] = fitted[winner], candidate_scores[winner]
            metrics[family] = candidates[winner]["metrics"]
            rules[family] = {"score_kind": "svm_decision_margin" if family == "svm" else "spoof_probability",
                             "threshold": threshold, "tie": "spoof"}
    leader = max(("logistic", "svm", "random_forest"), key=lambda name: metrics[name]["dev"]["balanced_accuracy"])
    details = {"metrics": metrics, "score_rules": rules, "search": searches, "selected": selected,
               "development_leader": leader, "candidate_count": sum(map(len, grid.values())),
               "selection_rule": "highest development balanced accuracy; exact ties keep first declared candidate",
               "family_tie_order": ["logistic", "svm", "random_forest"],
               "baseline_fit_details": {key: value for key, value in baseline.items() if key != "metrics"}}
    return models, scores, details


def run_comparison(index_path: str | Path, manifest: str | Path, cache_dir: str | Path,
                   output_dir: str | Path, config: ComparisonConfig = ComparisonConfig()) -> dict:
    target = Path(output_dir)
    if target.exists():
        raise ValueError("Comparison output directory already exists; choose a new run directory")
    x, y, index, audit = load_feature_partitions(index_path, manifest, cache_dir)
    models, scores, details = fit_comparison(x["train"], y["train"], x["dev"], y["dev"], config)
    report = {"schema_version": 1, "scope": "ASVspoof 2019 LA train/development subset; known attacks only",
              "positive_label": "spoof=1; bonafide=0", "config": asdict(config), **details,
              "source_manifest_sha256": _file_hash(Path(manifest)),
              "feature_index_sha256": _file_hash(Path(index_path)), "manifest_audit": audit,
              "feature_names": index["feature_names"], "feature_config": index["feature_config"],
              "feature_source_code_sha256": index["verified_feature_code_sha256"],
              "audio_config": index["audio_config"], "feature_runtime": index["runtime"],
              "model_runtime": {name: importlib.metadata.version(name)
                                for name in ("scikit-learn", "joblib", "numpy", "scipy", "threadpoolctl")},
              "trainer_source_sha256": {name: _file_hash(Path(__file__).with_name(name))
                                        for name in ("comparison.py", "baselines.py")},
              "model_definitions": {
                  "svm": "float64 cast + train-fitted StandardScaler + RBF SVC; probability fitting disabled by default; shrinking=True; cache_size=128 MiB",
                  "random_forest": "float64 cast + RandomForestClassifier; gini; sqrt max_features; bootstrap=True; n_jobs=1",
                  "logistic": "unchanged Day 5 float64/scaler/L2 logistic regression, lbfgs",
                  "majority": "most-frequent training class"},
              "model_input_dtype": "float64 (forest internally converts to float32)",
              "development_used_for_fit": False, "development_used_for_parameter_selection": True,
              "evaluation_audio_processed": False, "operating_threshold_selected": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-comparison-", dir=target.parent) as folder:
        staged = Path(folder)
        for name, model in models.items():
            joblib.dump(model, staged / f"{name}.joblib")
        np.savez_compressed(staged / "development_predictions.npz", labels=y["dev"],
                            **{name + "_scores": value["dev"] for name, value in scores.items()})
        report["local_artifact_sha256"] = {file.name: _file_hash(file) for file in staged.iterdir()}
        (staged / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.rename(target)
    return report
