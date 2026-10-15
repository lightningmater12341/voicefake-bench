"""Analytical metric cases, independent ranking references and artifact guards."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from voicefake.audio import _file_hash
from voicefake.baselines import classification_metrics
from voicefake.manifest import REQUIRED_COLUMNS, validate_manifest
from voicefake.metrics import (MetricConfig, binary_roc, roc_summary, score_development,
                               select_development_threshold, threshold_predictions)


class MetricTests(unittest.TestCase):
    def test_perfect_reversed_and_constant_rankings(self):
        labels = [0, 0, 1, 1]
        for scores, auc, eer in [([0, 1, 2, 3], 1, 0), ([3, 2, 1, 0], 0, 1), ([4, 4, 4, 4], .5, .5)]:
            with self.subTest(scores=scores):
                actual = roc_summary(labels, scores)
                self.assertAlmostEqual(actual["roc_auc"], auc)
                self.assertAlmostEqual(actual["eer"], eer)

    def test_mixed_tie_block_has_analytical_interpolated_eer(self):
        result = roc_summary([0, 0, 1, 1], [.8, .2, .8, .8])
        self.assertAlmostEqual(result["roc_auc"], .75)
        self.assertAlmostEqual(result["eer"], 1 / 3)
        self.assertAlmostEqual(result["eer_bracket"]["interpolation_weight"], 2 / 3)
        self.assertEqual(result["roc_points"], 3)

    def test_random_tied_roc_matches_sklearn_and_pairwise_auc(self):
        generator = np.random.default_rng(42)
        labels = np.repeat([0, 1], [17, 31])
        for _ in range(25):
            scores = generator.integers(-4, 5, len(labels)).astype(float)
            fpr, tpr, thresholds = binary_roc(labels, scores)
            expected = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
            for actual, reference in zip((fpr, tpr, thresholds), expected):
                np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-15)
            negative, positive = scores[labels == 0], scores[labels == 1]
            wins = positive[:, None] > negative
            ties = positive[:, None] == negative
            pairwise = float(np.mean(wins + .5 * ties))
            summary = roc_summary(labels, scores)
            self.assertAlmostEqual(summary["roc_auc"], pairwise)
            self.assertAlmostEqual(summary["roc_auc"], roc_auc_score(labels, scores))
            bracket = summary["eer_bracket"]
            weight = bracket["interpolation_weight"]
            fnr = bracket["lower_fnr"] + weight * (bracket["upper_fnr"] - bracket["lower_fnr"])
            self.assertAlmostEqual(summary["eer"], fnr)
            permutation = generator.permutation(len(labels))
            self.assertEqual(summary, roc_summary(labels[permutation], scores[permutation]))

    def test_invalid_metric_inputs_are_rejected(self):
        for labels, scores in [([0, 0], [1, 2]), ([0, 1], [1]), ([0, 2], [1, 2]),
                               ([0, 1], [np.nan, 1]), ([0, 1], [np.inf, 1]),
                               ([0, 1], [[1], [2]]), ([0, 1], ["1", "2"]), ([], [])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                roc_summary(labels, scores)

    def test_threshold_budget_and_tied_genuine_scores(self):
        point = select_development_threshold([.9, .8, .8, .1], MetricConfig(target_fpr=.5))
        self.assertEqual(point["false_positive_budget"], 2)
        self.assertEqual(point["genuine_cutoff_score"], .8)
        self.assertEqual(point["decision_rule"]["threshold"], np.nextafter(.8, np.inf))
        self.assertEqual(int(threshold_predictions([.9, .8, .8, .1], point["decision_rule"]).sum()), 1)
        self.assertEqual(threshold_predictions([point["decision_rule"]["threshold"]], point["decision_rule"])[0], 1)

    def test_decimal_budget_and_small_sample_resolution(self):
        point = select_development_threshold(np.arange(100), MetricConfig(target_fpr=.29))
        self.assertEqual(point["false_positive_budget"], 29)
        self.assertEqual(int(threshold_predictions(np.arange(100), point["decision_rule"]).sum()), 29)
        self.assertEqual(select_development_threshold(np.arange(99))["false_positive_budget"], 0)

    def test_threshold_is_most_permissive_feasible_rule(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([.9, .8, .8, .1, .95, .85, .8, .3])
        point = select_development_threshold(scores[labels == 0], MetricConfig(target_fpr=.5))
        predicted = threshold_predictions(scores, point["decision_rule"])
        self.assertEqual(int(predicted[labels == 0].sum()), 1)
        for threshold in np.r_[np.unique(scores), np.nextafter(np.unique(scores), np.inf)]:
            alternative = scores >= threshold
            if int(alternative[labels == 0].sum()) <= 2:
                self.assertLessEqual(int(alternative[labels == 1].sum()), int(predicted[labels == 1].sum()))
        # Adding arbitrary spoof scores cannot influence a genuine-only selection function.
        altered = scores.copy(); altered[labels == 1] += 1000
        self.assertEqual(point, select_development_threshold(altered[labels == 0], MetricConfig(target_fpr=.5)))

    def test_saved_rule_roundtrip_and_finite_extreme(self):
        for genuine in ([1, 1], [np.finfo(np.float64).max]):
            point = select_development_threshold(genuine, MetricConfig(target_fpr=0))
            restored = json.loads(json.dumps(point, allow_nan=False))
            self.assertEqual(point, restored)
            self.assertFalse(threshold_predictions(genuine, restored["decision_rule"]).any())
        self.assertEqual(point["decision_rule"], {"mode": "reject_all", "threshold": None})
        with self.assertRaises(ValueError):
            threshold_predictions([.1], {"mode": "score_at_least", "threshold": np.inf})

    def test_config_and_invalid_target(self):
        path = Path(__file__).resolve().parents[1] / "configs/metrics.json"
        self.assertEqual(MetricConfig.from_json(path), MetricConfig())
        for value in (-.1, 1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MetricConfig(target_fpr=value)

    def fixture(self, root):
        source = root / "comparison"; source.mkdir()
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        rows = []
        for split, values in (("train", [0, 1]), ("dev", labels)):
            for i, label in enumerate(values):
                rows.append(dict(audio_path=f"{split}/{i}.flac", label="spoof" if label else "bonafide",
                                 split=split, speaker_id=split + "_speaker", attack_id="A01" if label else "-"))
        manifest = root / "manifest.csv"
        self.write_manifest(manifest, rows)
        scores = {"majority": np.ones(6), "logistic": np.array([.7, .4, .2, .9, .6, .1]),
                  "svm": np.array([1, .5, -1, 2, .3, -.5]), "random_forest": np.array([.6, .3, .1, .8, .7, .4])}
        np.savez_compressed(source / "development_predictions.npz", labels=labels,
                            **{name + "_scores": values for name, values in scores.items()})
        report = {"source_manifest_sha256": _file_hash(manifest), "manifest_audit": validate_manifest(manifest),
                  "evaluation_audio_processed": False, "development_used_for_fit": False,
                  "positive_label": "spoof=1; bonafide=0", "feature_index_sha256": "0" * 64,
                  "score_rules": {}, "metrics": {}, "config": {"fixture": True},
                  "model_runtime": {"fixture": True}, "development_leader": "svm",
                  "local_artifact_sha256": {"development_predictions.npz": _file_hash(source / "development_predictions.npz")}}
        for name, values in scores.items():
            threshold = 0 if name == "svm" else .5
            report["score_rules"][name] = {"score_kind": "svm_decision_margin" if name == "svm" else "spoof_probability",
                                           "threshold": threshold, "tie": "spoof"}
            report["metrics"][name] = {"dev": classification_metrics(labels, (values >= threshold).astype(np.int8))}
            report["local_artifact_sha256"][name + ".joblib"] = "0" * 64
        (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return source, manifest, rows, report

    def write_manifest(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader(); writer.writerows(rows)

    def test_runner_repeats_without_loading_or_refitting_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, manifest, _, _ = self.fixture(root)
            with patch("joblib.load", side_effect=AssertionError("Models must not be loaded")):
                first = score_development(source, manifest, root / "metrics")
                second = score_development(source, manifest, root / "repeat")
            self.assertEqual(first, second)
            self.assertFalse(first["models_refitted"])
            self.assertFalse(first["evaluation_audio_processed"])
            self.assertFalse(first["experiment_frozen"])
            self.assertEqual(first["models"]["majority"]["operating_point"]["metrics"]["spoof_recall"], 0)
            with self.assertRaisesRegex(ValueError, "already exists"):
                score_development(source, manifest, root / "metrics")

    def test_runner_rejects_evaluation_membership_and_manifest_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, manifest, rows, _ = self.fixture(root)
            rows[-1]["speaker_id"] = "different_dev_speaker"
            self.write_manifest(manifest, rows)
            with self.assertRaisesRegex(ValueError, "Manifest differs"):
                score_development(source, manifest, root / "metrics")
            rows[-1].update(split="eval", speaker_id="evaluation_speaker")
            self.write_manifest(manifest, rows)
            with self.assertRaisesRegex(ValueError, "evaluation membership"):
                score_development(source, manifest, root / "metrics")
            self.assertFalse((root / "metrics").exists())

    def test_runner_rejects_score_corruption_and_label_misalignment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, manifest, _, report = self.fixture(root)
            path = source / "development_predictions.npz"
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
            arrays["labels"] = arrays["labels"][::-1]
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "checksum"):
                score_development(source, manifest, root / "metrics")
            report["local_artifact_sha256"][path.name] = _file_hash(path)
            (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest order"):
                score_development(source, manifest, root / "metrics")


if __name__ == "__main__":
    unittest.main()
