"""Math and cache checks use controlled fixtures, never detector-performance claims."""

import csv
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voicefake.audio import AudioConfig
from voicefake.features import FeatureConfig, extract_features, feature_names, mel_filterbank, mfcc_frames
from voicefake.feature_cache import cached_features, extract_manifest
from voicefake.manifest import REQUIRED_COLUMNS


def tone(frequency=440):
    return (0.2 * np.sin(2 * np.pi * frequency * np.arange(16000) / 16000)).astype(np.float32)


class FeatureTests(unittest.TestCase):
    def test_independent_dft_and_cosine_reference(self):
        cfg = FeatureConfig(frame_length=16, hop_length=8, n_fft=32, n_mels=4,
                            n_mfcc=3, fmin_hz=0, fmax_hz=8000)
        signal = np.random.default_rng(7).normal(size=24)
        actual = mfcc_frames(signal, 16000, cfg)
        mel_max = 2595 * math.log10(1 + 8000 / 700)
        edges = [700 * (10 ** (mel_max * i / 5 / 2595) - 1) for i in range(6)]
        expected = []
        for start in (0, 8):
            powers = []
            for k in range(17):
                value = sum(signal[start + n] * (0.5 - 0.5 * math.cos(2 * math.pi * n / 16))
                            * np.exp(-2j * math.pi * k * n / 32) for n in range(16))
                powers.append(abs(value) ** 2 / 32)
            logs = []
            for m in range(4):
                energy = 0.0
                for k, power in enumerate(powers):
                    frequency = k * 16000 / 32
                    weight = max(0, min((frequency - edges[m]) / (edges[m + 1] - edges[m]),
                                        (edges[m + 2] - frequency) / (edges[m + 2] - edges[m + 1])))
                    energy += power * weight
                logs.append(math.log(max(energy, cfg.log_floor)))
            expected.append([math.sqrt(1 / 4 if c == 0 else 2 / 4) * sum(
                logs[m] * math.cos(math.pi * c * (m + 0.5) / 4) for m in range(4)) for c in range(3)])
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_silent_frames_are_finite_with_known_c0(self):
        frames = mfcc_frames(np.zeros(400), 16000)
        self.assertAlmostEqual(frames[0, 0], math.sqrt(40) * math.log(1e-12))
        np.testing.assert_allclose(frames[0, 1:], 0, atol=1e-12)

    def test_amplitude_change_has_analytical_c0_shift(self):
        signal = np.random.default_rng(4).normal(size=3200) * 0.1
        first = mfcc_frames(signal, 16000)
        second = mfcc_frames(signal * 2, 16000)
        np.testing.assert_allclose(second[:, 0] - first[:, 0], 2 * math.log(2) * math.sqrt(40), atol=1e-11)
        np.testing.assert_allclose(second[:, 1:], first[:, 1:], atol=1e-11)

    def test_summary_dimensions_and_tail(self):
        vector, report = extract_features(tone()[:401], 16000)
        self.assertEqual(vector.shape, (40,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["unused_tail_samples"], 1)
        np.testing.assert_array_equal(vector[20:], 0)
        longer, details = extract_features(tone(), 16000)
        self.assertEqual(longer.shape, vector.shape)
        self.assertEqual(details["frame_count"], 98)
        self.assertTrue(np.isfinite(longer).all())
        self.assertEqual(len(set(feature_names())), 40)

    def test_invalid_inputs_and_rate_rejected(self):
        for signal in (np.ones(10), np.ones((400, 2)), np.full(400, np.nan)):
            with self.subTest(shape=signal.shape), self.assertRaises(ValueError):
                mfcc_frames(signal, 16000)
        with self.assertRaisesRegex(ValueError, "sample rate"):
            mfcc_frames(tone(), 8000)

    def test_configuration_rejects_invalid_or_empty_filters(self):
        for values in ({"n_fft": 100}, {"n_mfcc": 41}, {"fmax_hz": 9000},
                       {"log_floor": 0}, {"hop_length": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FeatureConfig(**values)
        with self.assertRaisesRegex(ValueError, "Empty mel"):
            mel_filterbank(FeatureConfig(fmin_hz=100, fmax_hz=101))

    def test_recipe_matches_defaults(self):
        path = Path(__file__).resolve().parents[1] / "configs/features.json"
        self.assertEqual(asdict(FeatureConfig.from_json(path)), asdict(FeatureConfig()))


class CacheTests(unittest.TestCase):
    def source(self, root, name="a.wav", frequency=440):
        path = root / name
        sf.write(path, tone(frequency), 16000, subtype="PCM_16")
        return path

    def manifest(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_hit_avoids_decode_and_is_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.source(root)
            first, report = cached_features(path, root / "cache")
            self.assertFalse(report["cache_hit"])
            with patch("voicefake.feature_cache.load_audio", side_effect=AssertionError("unexpected decode")):
                second, repeated = cached_features(path, root / "cache")
            self.assertTrue(repeated["cache_hit"])
            np.testing.assert_array_equal(first, second)
            self.assertEqual(report["vector_sha256"], repeated["vector_sha256"])

    def test_source_and_configuration_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.source(root)
            _, first = cached_features(path, root / "cache")
            _, cropped = cached_features(path, root / "cache", AudioConfig(max_duration_seconds=0.5))
            short, reduced = cached_features(path, root / "cache", feature_config=FeatureConfig(n_mfcc=12))
            self.assertEqual(short.shape, (24,))
            self.source(root, frequency=880)
            _, changed = cached_features(path, root / "cache")
            self.assertEqual(len({r["cache_key"] for r in (first, cropped, reduced, changed)}), 4)
            self.assertFalse(changed["cache_hit"])

    def test_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.source(root)
            _, report = cached_features(path, root / "cache")
            file = root / "cache" / report["cache_key"] / "features.npy"
            np.save(file, np.ones(40, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Invalid cache"):
                cached_features(path, root / "cache")

    def test_manifest_labels_separate_from_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source(root, "a.wav", 440)
            self.source(root, "b.wav", 880)
            rows = [{"audio_path": name, "label": "bonafide", "split": split,
                     "speaker_id": speaker, "attack_id": "-"}
                    for name, split, speaker in (("a.wav", "train", "s1"), ("b.wav", "dev", "s2"))]
            manifest = root / "manifest.csv"
            self.manifest(manifest, rows)
            first = extract_manifest(manifest, root, root / "cache", root / "index-a.json")
            self.assertEqual(first["feature_shape"], [2, 40])
            self.assertFalse(first["learned_preprocessing_fitted"])
            rows[1].update(label="spoof", attack_id="A01")
            self.manifest(manifest, rows)
            second = extract_manifest(manifest, root, root / "cache", root / "index-b.json")
            self.assertEqual(second["cache_hits"], 2)
            self.assertEqual([r["vector_sha256"] for r in first["records"]],
                             [r["vector_sha256"] for r in second["records"]])
            with self.assertRaisesRegex(ValueError, "already exists"):
                extract_manifest(manifest, root, root / "cache", root / "index-a.json")

    def test_evaluation_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source(root)
            manifest = root / "manifest.csv"
            self.manifest(manifest, [{"audio_path": "a.wav", "label": "bonafide",
                                     "split": "eval", "speaker_id": "s1", "attack_id": "-"}])
            with self.assertRaisesRegex(ValueError, "after experiment freeze"):
                extract_manifest(manifest, root, root / "cache", root / "index.json")
            self.assertFalse((root / "index.json").exists())


if __name__ == "__main__":
    unittest.main()
