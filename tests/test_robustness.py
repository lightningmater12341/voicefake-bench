"""Controlled score fixtures test pairing, clustering and failures; never benchmark speech."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voicefake.robustness import CONDITIONS, run_robustness, speaker_draws, summarize_paired


def fixture():
    rows = [{"audio_path": f"dev/{i}.flac", "split": "dev", "speaker_id": speaker,
             "label": label, "attack_id": "-" if label == "bonafide" else "A01"}
            for i, (speaker, label) in enumerate([("a", "bonafide"), ("a", "spoof"),
                                                 ("b", "bonafide"), ("b", "spoof")])]
    scores = {"fixed": {condition: np.array([0., 1., 0., 1.]) for condition in CONDITIONS}}
    definitions = {"fixed": {"decision_rule": {"mode": "score_at_least", "threshold": .5}}}
    plan = {"method": "paired speaker-block bootstrap", "replicates": 20, "seed": 42,
            "confidence": .95, "interval": "percentile; numpy quantile method=linear",
            "threshold_reselection": False, "invalid_draws": "exclude missing class; never replace"}
    return rows, scores, definitions, plan


class RobustnessTests(unittest.TestCase):
    def test_draws_keep_whole_blocks_and_duplicate_speakers(self):
        rows, _, _, _ = fixture()
        draws = list(speaker_draws(rows, 40, 42))
        self.assertTrue(any(len(set(draw)) == 2 for draw in draws))
        for draw in draws:
            self.assertEqual(len(draw), 4)
            for index in (0, 2):
                self.assertEqual(int(np.sum(draw == index)), int(np.sum(draw == index+1)))
        replay = list(speaker_draws(rows, 40, 42))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(draws, replay)))
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(draws, speaker_draws(rows, 40, 43))))

    def test_identical_conditions_have_exact_zero_paired_intervals(self):
        rows, scores, rules, plan = fixture()
        original = copy.deepcopy(rules)
        result = summarize_paired(rows, scores, rules, plan)
        self.assertEqual(result["bootstrap"]["valid_draws"], 20)
        self.assertEqual(result["bootstrap"]["invalid_draws"], 0)
        for condition in CONDITIONS:
            for stats in result["models"]["fixed"][condition]["statistics"].values():
                self.assertEqual(stats["difference_from_clean"], 0.)
                self.assertEqual(stats["paired_difference_interval_95"], [0., 0.])
        self.assertEqual(rules, original)
        self.assertEqual(result, summarize_paired(rows, scores, rules, plan))

    def test_known_reversal_has_exact_paired_changes_and_attack_counts(self):
        rows, scores, rules, plan = fixture()
        scores["fixed"]["noise_10db"] = np.array([1., 0., 1., 0.])
        report = summarize_paired(rows, scores, rules, plan)["models"]["fixed"]["noise_10db"]
        self.assertEqual(report["confusion_matrix"], [[0, 2], [2, 0]])
        self.assertEqual(report["statistics"]["roc_auc"]["paired_difference_interval_95"], [-1., -1.])
        self.assertEqual(report["statistics"]["false_positive_rate"]["paired_difference_interval_95"], [1., 1.])
        self.assertEqual(report["statistics"]["spoof_recall"]["paired_difference_interval_95"], [-1., -1.])
        self.assertEqual(report["per_attack"]["A01"], {"spoof_count": 2, "detected": 0, "spoof_recall": 0.})
        self.assertEqual(report["paired_decision_changes"]["genuine_new_false_alarms"], 2)
        self.assertEqual(report["paired_decision_changes"]["spoofs_no_longer_detected"], 2)

    def test_invalid_class_draws_are_counted_and_not_replaced(self):
        rows, scores, rules, plan = fixture()
        rows[1]["speaker_id"], rows[2]["speaker_id"] = "b", "a"
        # Speaker a contains only genuine; b contains only spoof.
        known_draws = [np.array([0,2,0,2]), np.arange(4), np.array([1,3,1,3])]
        with patch("voicefake.robustness.speaker_draws", return_value=iter(known_draws)):
            result = summarize_paired(rows, scores, rules, plan)
        self.assertEqual(result["bootstrap"]["valid_draws"], 1)
        self.assertEqual(result["bootstrap"]["invalid_draws"], 2)
        self.assertEqual(result["bootstrap"]["requested_draws"], 3)
        with patch("voicefake.robustness.speaker_draws", return_value=iter(known_draws[:1])):
            result = summarize_paired(rows, scores, rules, plan)
        self.assertEqual(result["bootstrap"]["valid_draws"], 0)
        self.assertIsNone(result["models"]["fixed"]["clean"]["statistics"]["eer"]["interval_95"])

    def test_percentile_uses_linear_quantiles_of_paired_differences(self):
        rows, scores, rules, plan = fixture()
        scores["fixed"]["mp3_32kbps"] = np.array([0., 1., 0., 0.])
        known_draws = [np.array([0,1,0,1]), np.array([2,3,2,3])]
        with patch("voicefake.robustness.speaker_draws", return_value=iter(known_draws)):
            result = summarize_paired(rows, scores, rules, plan)
        stats=result["models"]["fixed"]["mp3_32kbps"]["statistics"]["spoof_recall"]
        np.testing.assert_allclose(stats["interval_95"], [.025, .975], rtol=0, atol=1e-15)
        np.testing.assert_allclose(stats["paired_difference_interval_95"], [-.975, -.025], rtol=0, atol=1e-15)

    def test_fixed_threshold_ties_and_majority_do_not_get_reselected(self):
        rows, scores, rules, plan = fixture()
        for condition in CONDITIONS:
            scores["fixed"][condition] = np.full(4, .5)
        result = summarize_paired(rows, scores, rules, plan)
        self.assertEqual(result["models"]["fixed"]["clean"]["confusion_matrix"], [[0,2],[0,2]])
        rules["fixed"]["decision_rule"]["threshold"] = float(np.nextafter(.5, 1.))
        result = summarize_paired(rows, scores, rules, plan)
        self.assertEqual(result["models"]["fixed"]["clean"]["confusion_matrix"], [[2,0],[2,0]])

    def test_rejects_unpaired_nonfinite_or_evaluation_inputs(self):
        for mutation in ("missing", "length", "nan", "eval", "duplicate", "label", "one_class", "retune"):
            with self.subTest(mutation=mutation):
                rows, scores, rules, plan = fixture()
                if mutation == "missing": scores["fixed"].pop("clean")
                if mutation == "length": scores["fixed"]["clean"] = np.array([0.])
                if mutation == "nan": scores["fixed"]["clean"][0] = np.nan
                if mutation == "eval": rows[0]["split"] = "eval"
                if mutation == "duplicate": rows[0]["audio_path"] = rows[1]["audio_path"]
                if mutation == "label": rows[0]["label"] = "unknown"
                if mutation == "one_class":
                    for row in rows: row["label"] = "spoof"
                if mutation == "retune": plan["threshold_reselection"] = True
                with self.assertRaises(ValueError): summarize_paired(rows, scores, rules, plan)

    def test_runner_checks_freeze_before_decoding_or_loading_models(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("voicefake.robustness.verify_freeze", side_effect=ValueError("freeze drift")), patch("voicefake.robustness.load_audio") as decode, patch("voicefake.robustness.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "freeze drift"):
                    run_robustness(folder)
                decode.assert_not_called(); load.assert_not_called()

    def test_runner_refuses_output_overwrite_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/"artifacts/existing").mkdir(parents=True)
            for output in ("../outside", "docs/run", "artifacts/existing"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    run_robustness(root, output)

    def test_draw_validation(self):
        rows, _, _, _ = fixture()
        for replicates, seed in ((0,42), (True,42), (1,-1)):
            with self.assertRaises(ValueError): list(speaker_draws(rows,replicates,seed))
        for row in rows: row["speaker_id"] = "same"
        with self.assertRaises(ValueError): list(speaker_draws(rows,1,42))


if __name__ == "__main__":
    unittest.main()
