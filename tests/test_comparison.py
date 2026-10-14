"""Controlled fixtures test search boundaries; real results live in docs."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import test_baselines as fixtures

from voicefake.baselines import load_feature_partitions
from voicefake.comparison import ComparisonConfig, fit_comparison, model_scores, run_comparison, select_candidate


def small_config():
    return replace(ComparisonConfig(), svm_c=(1.0,), svm_gamma=("scale",),
                   forest_trees=12, forest_depth=(None,), forest_min_leaf=(1,))


class ComparisonTests(unittest.TestCase):
    def test_declared_recipe_and_ten_candidate_budget(self):
        path = Path(__file__).resolve().parents[1] / "configs/comparison.json"
        actual = json.loads(json.dumps(asdict(ComparisonConfig())))
        self.assertEqual(actual, asdict(ComparisonConfig.from_json(path)))
        config = ComparisonConfig()
        self.assertEqual(len(config.svm_c) * len(config.svm_gamma) +
                         len(config.forest_depth) * len(config.forest_min_leaf), 10)

    def test_invalid_and_oversized_grids_rejected(self):
        cases = ({"svm_c": ()}, {"svm_c": (1, 1)}, {"svm_c": (float("nan"),)},
                 {"svm_gamma": ("auto",)}, {"forest_depth": (0,)},
                 {"forest_trees": 201}, {"forest_min_leaf": (True,)}, {"svm_max_iter": -1},
                 {"svm_c": tuple(range(1, 10)), "svm_gamma": ("scale", .1)})
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                ComparisonConfig(**case)

    def test_selection_uses_development_balanced_accuracy_and_first_tie(self):
        candidates = [{"metrics": {"train": {"balanced_accuracy": train},
                                  "dev": {"balanced_accuracy": dev, "accuracy": raw}}}
                      for train, dev, raw in [(1, .6, .99), (.8, .7, .5), (1, .7, 1)]]
        self.assertEqual(select_candidate(candidates), 1)

    def test_svm_train_scaler_and_fit_ignore_development_shift(self):
        train, labels, dev, dev_labels = fixtures.BaselineTests().arrays()
        models, scores, report = fit_comparison(train, labels, dev, dev_labels, small_config())
        shifted, _, _ = fit_comparison(train, labels, dev + 1000, dev_labels, small_config())
        svm = models["svm"]
        np.testing.assert_allclose(svm.named_steps["scaler"].mean_, train.mean(axis=0), atol=1e-12)
        np.testing.assert_array_equal(svm.named_steps["classifier"].support_vectors_,
                                     shifted["svm"].named_steps["classifier"].support_vectors_)
        np.testing.assert_array_equal(svm.named_steps["classifier"].dual_coef_,
                                     shifted["svm"].named_steps["classifier"].dual_coef_)
        self.assertEqual(report["search"]["svm"][0]["scaler_fit_rows"], 60)
        self.assertEqual(report["score_rules"]["svm"]["threshold"], 0)
        np.testing.assert_array_equal(scores["svm"]["dev"] >= 0, svm.predict(dev) == 1)
        self.assertGreater(scores["svm"]["dev"][dev_labels == 1].mean(),
                           scores["svm"]["dev"][dev_labels == 0].mean())
        self.assertEqual(svm.named_steps["classifier"].probA_.size, 0)
        self.assertNotIn("scaler", models["random_forest"].named_steps)

    def test_seeded_forest_and_svm_repeat_exactly(self):
        arrays = fixtures.BaselineTests().arrays()
        _, first, details = fit_comparison(*arrays, small_config())
        _, second, repeated = fit_comparison(*arrays, small_config())
        self.assertEqual(details, repeated)
        for name in first:
            for split in first[name]:
                np.testing.assert_array_equal(first[name][split], second[name][split])

    def test_nonconvergent_svm_does_not_publish(self):
        with self.assertRaisesRegex(ValueError, "converge"):
            fit_comparison(*fixtures.BaselineTests().arrays(), replace(small_config(), svm_max_iter=1))

    def test_selected_artifacts_reload_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, index, _ = fixtures.BaselineTests().dataset(root)
            report = run_comparison(index, manifest, root / "cache", root / "run", small_config())
            features, _, _, _ = load_feature_partitions(index, manifest, root / "cache")
            with np.load(root / "run/development_predictions.npz", allow_pickle=False) as saved:
                for name in report["metrics"]:
                    model = joblib.load(root / "run" / (name + ".joblib"))
                    kind = "margin" if name == "svm" else "probability"
                    np.testing.assert_allclose(model_scores(model, features["dev"], kind),
                                               saved[name + "_scores"], rtol=0, atol=1e-12)
            self.assertFalse(report["development_used_for_fit"])
            self.assertFalse(report["evaluation_audio_processed"])
            self.assertTrue(report["development_used_for_parameter_selection"])
            with patch("voicefake.comparison.load_feature_partitions") as loader:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    run_comparison(index, manifest, root / "cache", root / "run")
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
