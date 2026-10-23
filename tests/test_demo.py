"""Inference boundary fixtures; synthetic signals and fake models are not benchmark results."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voicefake.audio import AudioConfig, _file_hash, load_audio
from voicefake.audit import waveform_digest
from voicefake.demo import _evaluation_guard, demo_audio


class FixtureModel:
    classes_ = np.array([0, 1])
    def __init__(self, margin):
        self.margin = margin
    def decision_function(self, features):
        if features.shape != (1, 40) or features.dtype != np.float64:
            raise ValueError("Unexpected model input")
        return np.array([self.margin])


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/audio.json").write_text("{}", encoding="utf-8")
        (self.root / "configs/features.json").write_text("{}", encoding="utf-8")
        self.details = {"artifact": "artifacts/svm.joblib", "sha256": "model-fixture",
                        "decision_rule": {"mode": "score_at_least", "threshold": 0.25}}
        self.spec = {"primary_model": "svm", "models": {"svm": self.details}}
        self.save_spec()
        self.path = self.root / "input.wav"
        sf.write(self.path, .1 * np.sin(2*np.pi*440*np.arange(8000)/16000), 16000, subtype="FLOAT")
        self.verify = patch("voicefake.demo.verify_freeze", return_value={"lock_sha256": "fixture-lock"})
        self.guard = patch("voicefake.demo._evaluation_guard", return_value=(set(), set(), set()))
        self.verify.start(); self.guard.start()
        self.addCleanup(self.verify.stop); self.addCleanup(self.guard.stop)

    def save_spec(self):
        (self.root / "configs/frozen-experiment.json").write_text(json.dumps(self.spec), encoding="utf-8")

    def run_demo(self, margin=.5):
        with patch("voicefake.demo.joblib.load", return_value=FixtureModel(margin)):
            return demo_audio(self.path, self.root)

    def test_threshold_tie_flags_and_below_threshold_does_not_verify_origin(self):
        self.assertEqual(self.run_demo(.25)["decision"], "flagged_as_spoof")
        result = self.run_demo(np.nextafter(.25, -np.inf))
        self.assertEqual(result["decision"], "not_flagged_as_spoof")
        self.assertFalse(result["score_is_probability"])
        self.assertNotIn(str(self.root), json.dumps(result))
        json.dumps(result, allow_nan=False)

    def test_demo_writes_nothing_and_reuses_audio_recipe(self):
        before = {p.relative_to(self.root): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        result = self.run_demo()
        after = {p.relative_to(self.root): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["input"]["output_frames"], 8000)
        self.assertEqual(result["feature_count"], 40)
        self.assertFalse(result["models_refitted"])
        self.assertFalse(result["threshold_changed"])

    def test_freeze_failure_precedes_decode_and_model_deserialization(self):
        with patch("voicefake.demo.verify_freeze", side_effect=ValueError("freeze drift")), patch("voicefake.demo.load_audio") as decode, patch("voicefake.demo.joblib.load") as model:
            with self.assertRaisesRegex(ValueError, "freeze drift"): demo_audio(self.path, self.root)
            decode.assert_not_called(); model.assert_not_called()

    def test_evaluation_path_and_encoded_copy_refused_before_decode(self):
        for guard in (({self.path.resolve()}, set(), set()), (set(), {_file_hash(self.path)}, set())):
            with patch("voicefake.demo._evaluation_guard", return_value=guard), patch("voicefake.demo.load_audio") as decode, patch("voicefake.demo.joblib.load") as model:
                with self.assertRaisesRegex(ValueError, "evaluation"): demo_audio(self.path, self.root)
                decode.assert_not_called(); model.assert_not_called()

    def test_reencoded_exact_waveform_refused_before_model_load(self):
        waveform, _ = load_audio(self.path)
        digest = waveform_digest(waveform, 16000, "<f4")
        with patch("voicefake.demo._evaluation_guard", return_value=(set(), set(), {digest})), patch("voicefake.demo.joblib.load") as model:
            with self.assertRaisesRegex(ValueError, "evaluation content"): demo_audio(self.path, self.root)
            model.assert_not_called()

    def test_bad_audio_never_loads_model(self):
        for waveform in (np.zeros(8000), np.ones(100)):
            sf.write(self.path, waveform, 16000, subtype="FLOAT")
            with patch("voicefake.demo.joblib.load") as model:
                with self.assertRaises(ValueError): demo_audio(self.path, self.root)
                model.assert_not_called()

    def test_stereo_resampling_and_center_crop_are_reported(self):
        signal = .1*np.sin(2*np.pi*440*np.arange(11*8000)/8000)
        sf.write(self.path, np.column_stack([signal, signal]), 8000, subtype="FLOAT")
        result = self.run_demo()
        self.assertTrue(result["input"]["resampled"])
        self.assertTrue(result["input"]["cropped"])
        self.assertEqual(result["input"]["output_frames"], 160000)

    def test_nonfinite_scores_fail(self):
        for margin in (np.nan, np.inf):
            with self.assertRaises(ValueError): self.run_demo(margin)

    def test_different_primary_is_not_silently_selected(self):
        self.spec["primary_model"] = "random_forest"; self.save_spec()
        with patch("voicefake.demo.joblib.load") as model:
            with self.assertRaisesRegex(ValueError, "primary SVM"): demo_audio(self.path, self.root)
            model.assert_not_called()

    def test_guard_checks_evidence_hash_and_manifest_membership(self):
        (self.root / "docs").mkdir()
        (self.root / "artifacts/evaluation-audit/day13").mkdir(parents=True)
        row = {"audio_path": "eval/example.flac", "label": "spoof", "split": "eval",
               "speaker_id": "s1", "attack_id": "A07"}
        manifest = self.root / "eval.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
        fingerprints = self.root / "artifacts/evaluation-audit/day13/recordings.json"
        records = [{**row, "encoded_sha256": "encoded", "processed_sha256": "processed"}]
        fingerprints.write_text(json.dumps(records), encoding="utf-8")
        audit = {"status": "pass", "frozen_lock_sha256": "fixture-lock",
                 "source_eval_manifest_sha256": _file_hash(manifest),
                 "local_artifact_sha256": {"recordings.json": _file_hash(fingerprints)}}
        audit_path = self.root / "docs/EVALUATION_CONTENT_AUDIT.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        spec = {"dataset": {"evaluation_manifest": "eval.csv"}}
        paths, encoded, processed = _evaluation_guard(self.root, spec, "fixture-lock")
        self.assertEqual(encoded, {"encoded"}); self.assertEqual(processed, {"processed"})
        self.assertEqual(len(paths), 1)
        records[0]["speaker_id"] = "changed"
        fingerprints.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprints changed"):
            _evaluation_guard(self.root, spec, "fixture-lock")
        audit["local_artifact_sha256"]["recordings.json"] = _file_hash(fingerprints)
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "membership differs"):
            _evaluation_guard(self.root, spec, "fixture-lock")


if __name__ == "__main__":
    unittest.main()
