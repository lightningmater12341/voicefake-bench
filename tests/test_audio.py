"""Controlled audio fixtures test signal handling, not fake-voice detection."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from voicefake.audio import AudioConfig, load_audio, process_samples


def tone(rate=16000, seconds=1.0, frequency=440):
    return (0.25 * np.sin(2 * np.pi * frequency * np.arange(round(rate * seconds)) / rate)).astype(np.float32)


class AudioTests(unittest.TestCase):
    def test_native_mono_preserves_amplitude_and_silence(self):
        signal = tone()
        signal[:1000] = 0
        output, report = process_samples(signal, 16000)
        np.testing.assert_array_equal(output, signal)
        self.assertFalse(report["resampled"])
        self.assertFalse(report["padded"])
        self.assertEqual(output.dtype, np.float32)

    def test_stereo_channel_mean(self):
        signal = tone()
        output, report = process_samples(np.column_stack([signal, signal * 0.5]), 16000)
        np.testing.assert_allclose(output, signal * 0.75, atol=1e-7)
        self.assertEqual(report["source_channels"], 2)

    def test_resampling_duration_and_tone_frequency(self):
        output, report = process_samples(tone(rate=44100), 44100)
        self.assertEqual(len(output), 16000)
        self.assertEqual(report["resample_ratio"], {"up": 160, "down": 441})
        peak = np.argmax(np.abs(np.fft.rfft(output)))
        self.assertEqual(peak, 440)

    def test_downsampling_suppresses_above_nyquist_tone(self):
        output, _ = process_samples(tone(rate=48000, frequency=12000), 48000)
        self.assertLess(np.sqrt(np.mean(output[200:-200] ** 2)), 0.002)

    def test_center_crop_no_padding(self):
        signal = tone(seconds=3)
        output, report = process_samples(signal, 16000, AudioConfig(max_duration_seconds=1.0))
        np.testing.assert_array_equal(output, signal[16000:32000])
        self.assertTrue(report["cropped"])
        self.assertEqual(report["crop_start_frame"], 16000)
        short, details = process_samples(tone(seconds=0.5), 16000)
        self.assertEqual(len(short), 8000)
        self.assertFalse(details["padded"])

    def test_invalid_or_zero_energy_audio_rejected(self):
        invalid = [np.array([], dtype=np.float32), np.zeros(16000),
                   np.full(16000, np.nan), np.ones((2, 2, 2)), tone(seconds=0.1),
                   np.column_stack([tone(), -tone()])]
        for signal in invalid:
            with self.subTest(shape=signal.shape), self.assertRaises(ValueError):
                process_samples(signal, 16000)

    def test_source_duration_guard(self):
        with self.assertRaisesRegex(ValueError, "limits"):
            process_samples(tone(seconds=2), 16000,
                            AudioConfig(max_duration_seconds=1, max_source_duration_seconds=1))

    def test_invalid_config_rejected(self):
        configs = [{"target_sample_rate": 0}, {"min_duration_seconds": float("nan")},
                   {"max_duration_seconds": 0.1}, {"schema_version": 2}]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                AudioConfig(**config)

    def test_flac_decode_hash_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.flac"
            sf.write(path, tone(), 16000, subtype="PCM_16")
            output, first = load_audio(path)
            repeated, second = load_audio(path)
            np.testing.assert_array_equal(output, repeated)
            self.assertEqual(first, second)
            self.assertEqual(first["source_encoding"], {"format": "FLAC", "subtype": "PCM_16"})
            np.testing.assert_allclose(output, tone(), atol=1 / 32768)

    def test_unreadable_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "not_audio.flac"
            path.write_text("not an audio stream")
            with self.assertRaisesRegex(ValueError, "Could not decode"):
                load_audio(path)

    def test_config_recipe_matches_defaults(self):
        recipe = Path(__file__).resolve().parents[1] / "configs/audio.json"
        self.assertEqual(asdict(AudioConfig.from_json(recipe)), asdict(AudioConfig()))

    def test_unknown_config_fields_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"trim_silence": True}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                AudioConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
