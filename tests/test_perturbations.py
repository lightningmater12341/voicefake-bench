"""Perturbation properties use controlled signals; no detector scores are invented."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import test_audit as fixtures

from voicefake.audit import audit_recordings
from voicefake.audio import _file_hash
from voicefake.perturbations import PerturbationConfig, apply_perturbation, audit_perturbations


def tone(frequency=1000, frames=16000, amplitude=.2):
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(frames) / 16000)).astype(np.float32)


class PerturbationTests(unittest.TestCase):
    def test_recipe_and_invalid_configuration(self):
        path = Path(__file__).resolve().parents[1] / "configs/perturbations.json"
        self.assertEqual(PerturbationConfig.from_json(path), PerturbationConfig())
        self.assertEqual(PerturbationConfig().conditions, ("noise_20db", "noise_10db", "bandwidth_8khz"))
        for values in ({"seed": -1}, {"sample_rate": 8000}, {"bandwidth_rate": 4000},
                       {"noise_snrs_db": []}, {"noise_snrs_db": [10, 10]},
                       {"noise_snrs_db": [float("nan")]}, {"noise_snrs_db": [True]},
                       {"noise_snrs_db": [20.1]}, {"noise_snrs_db": [-1]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PerturbationConfig(**values)

    def test_measured_snr_matches_independent_norm_calculation(self):
        clean = tone()
        for snr in (20, 10):
            output, details = apply_perturbation(clean, f"noise_{snr}db")
            measured = 20 * math.log10(np.linalg.norm(clean.astype(float)) /
                                       np.linalg.norm(output.astype(float) - clean.astype(float)))
            self.assertAlmostEqual(measured, snr, delta=.00001)
            self.assertAlmostEqual(measured, details["measured_snr_db"], places=10)

    def test_noise_levels_share_realization_with_different_amplitude(self):
        clean = tone()
        high, high_details = apply_perturbation(clean, "noise_20db")
        low, low_details = apply_perturbation(clean, "noise_10db")
        self.assertEqual(high_details["noise_seed_sha256"], low_details["noise_seed_sha256"])
        np.testing.assert_allclose(low.astype(float) - clean, math.sqrt(10) * (high.astype(float) - clean), rtol=1e-4, atol=1e-7)

    def test_repeat_order_seed_and_global_rng_independence(self):
        clean = tone(); before = np.random.get_state()
        first, details = apply_perturbation(clean, "noise_20db")
        apply_perturbation(tone(600), "noise_10db")
        repeated, again = apply_perturbation(clean.copy(), "noise_20db")
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(details, again)
        changed, _ = apply_perturbation(clean, "noise_20db", replace(PerturbationConfig(), seed=43))
        self.assertFalse(np.array_equal(first, changed))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(before[1], after[1])

    def test_clean_input_is_never_mutated_and_shapes_are_preserved(self):
        clean = tone(frames=8001); original = clean.copy(); clean.setflags(write=False)
        for condition in PerturbationConfig().conditions:
            output, details = apply_perturbation(clean, condition)
            self.assertEqual(output.shape, clean.shape)
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            self.assertEqual(details["sample_rate"], 16000)
            np.testing.assert_array_equal(clean, original)
        self.assertEqual(details["intermediate_frames"], 4001)
        self.assertEqual(details["right_tail_samples_removed"], 1)

    def test_noise_does_not_clip_or_peak_normalize(self):
        clean = np.full(16000, .99, dtype=np.float32)
        output, details = apply_perturbation(clean, "noise_10db")
        self.assertGreater(float(output.max()), 1)
        self.assertGreater(details["above_unit_samples"], 0)
        self.assertFalse(details["clipping_applied"])
        self.assertFalse(details["gain_normalization_applied"])

    def test_bandwidth_preserves_low_tone_and_suppresses_high_tone(self):
        low, _ = apply_perturbation(tone(1000), "bandwidth_8khz")
        high, _ = apply_perturbation(tone(6000), "bandwidth_8khz")
        reference = np.sqrt(np.mean(tone(1000)[200:-200].astype(float) ** 2))
        low_rms = np.sqrt(np.mean(low[200:-200].astype(float) ** 2))
        high_rms = np.sqrt(np.mean(high[200:-200].astype(float) ** 2))
        self.assertGreater(low_rms / reference, .99)
        self.assertLess(low_rms / reference, 1.01)
        self.assertLess(high_rms / reference, .003)

    def test_invalid_audio_or_condition_is_rejected(self):
        for values in (np.zeros(100, dtype=np.float32), np.ones(3, dtype=np.float32),
                       tone().astype(np.float64), tone().reshape(-1, 1), np.array([1, 2, np.nan, 4], dtype=np.float32)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                apply_perturbation(values, "noise_20db")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            apply_perturbation(tone(), "unsupported")

    def fixture(self, root):
        arguments, rows = fixtures.AuditTests().fixture(root)
        audit_recordings(**arguments)
        return {"manifest": arguments["manifest"], "audio_root": arguments["audio_root"],
                "audit_report": root / "audit/report.json", "recordings": root / "audit/recordings.json",
                "output_dir": root / "perturbations"}, rows

    def test_property_audit_replays_preserves_membership_and_writes_no_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); arguments, rows = self.fixture(root)
            first = audit_perturbations(**arguments)
            again = audit_perturbations(**{**arguments, "output_dir": root / "repeat"})
            self.assertEqual(first, again)
            self.assertEqual(first["transforms_checked"], 12)
            self.assertTrue(first["partition_membership_preserved"])
            for flag in ("evaluation_audio_processed", "models_refitted", "thresholds_changed",
                         "training_augmentation_performed", "detector_scores_computed", "waveform_files_written"):
                self.assertFalse(first[flag])
            self.assertEqual({path.name for path in arguments["output_dir"].iterdir()}, {"transforms.json", "report.json"})
            stored = json.loads((arguments["output_dir"] / "transforms.json").read_text(encoding="utf-8"))
            self.assertEqual([{name: row[name] for name in rows[0]} for row in stored], rows)
            self.assertEqual(_file_hash(arguments["output_dir"] / "transforms.json"), first["local_artifact_sha256"]["transforms.json"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                audit_perturbations(**arguments)

    def test_changed_recording_evidence_and_dirty_audit_are_rejected_before_audio(self):
        for mode in ("evidence", "dirty"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); arguments, _ = self.fixture(root)
                if mode == "evidence":
                    arguments["recordings"].write_text("[]", encoding="utf-8")
                else:
                    prior = json.loads(arguments["audit_report"].read_text(encoding="utf-8")); prior["status"] = "review_required"
                    arguments["audit_report"].write_text(json.dumps(prior), encoding="utf-8")
                with patch("voicefake.perturbations.load_audio") as loader:
                    with self.assertRaises(ValueError):
                        audit_perturbations(**arguments)
                    loader.assert_not_called()

    def test_evaluation_membership_is_refused_before_decoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); arguments, rows = self.fixture(root)
            rows.append(dict(audio_path="eval/example.flac",label="bonafide",split="eval",speaker_id="other",attack_id="-"))
            fixtures.AuditTests().write_manifest(arguments["manifest"], rows)
            with patch("voicefake.perturbations.load_audio") as loader:
                with self.assertRaisesRegex(ValueError, "evaluation audio"):
                    audit_perturbations(**arguments)
                loader.assert_not_called()

    def test_changed_clean_audio_is_rejected_and_no_completed_output_remains(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); arguments, rows = self.fixture(root)
            sf.write(arguments["audio_root"] / rows[0]["audio_path"], tone(1300,frames=8000), 16000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                audit_perturbations(**arguments)
            self.assertFalse(arguments["output_dir"].exists())


if __name__ == "__main__":
    unittest.main()
