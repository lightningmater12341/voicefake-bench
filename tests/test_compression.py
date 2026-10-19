"""Actual codec checks on controlled fixtures, not detector-performance tests."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_audit as fixtures

from voicefake.audit import audit_recordings
from voicefake.compression import CompressionConfig, audit_compression, codec_identity, mp3_roundtrip


HAS_CODEC = importlib.util.find_spec("imageio_ffmpeg") is not None


def tone(n=8001, amplitude=.2):
    return (amplitude * np.sin(2 * np.pi * 1000 * np.arange(n) / 16000)).astype(np.float32)


class CompressionTests(unittest.TestCase):
    def test_fixed_recipe_and_invalid_settings(self):
        path = Path(__file__).resolve().parents[1] / "configs/compression.json"
        self.assertEqual(CompressionConfig.from_json(path), CompressionConfig())
        for values in ({"headroom_gain": 1}, {"bitrate_kbps": 64}, {"sample_rate": 8000},
                       {"algorithm_quality": True}, {"schema_version": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CompressionConfig(**values)

    def test_invalid_input_and_headroom_fail_before_codec(self):
        for wave in (np.zeros(4000,dtype=np.float32), np.ones(4000,dtype=float),
                     tone(3999), tone().reshape(-1,1), np.full(4000,2,dtype=np.float32)):
            with patch("voicefake.compression.codec_identity") as identity:
                with self.assertRaises(ValueError):
                    mp3_roundtrip(wave)
                identity.assert_not_called()

    def test_unexpected_decoder_length_is_not_repaired(self):
        def fake(arguments, payload=None):
            if arguments[-1] == "pipe:1":
                return b"\0" * 4
            Path(arguments[-1]).write_bytes(b"fixture")
            return b""
        with patch("voicefake.compression.codec_identity", return_value={}), patch("voicefake.compression._run", side_effect=fake):
            with self.assertRaisesRegex(ValueError,"declared codec-input length"):
                mp3_roundtrip(tone())

    @unittest.skipUnless(HAS_CODEC,"Install the optional compression group")
    def test_even_odd_and_regression_boundary_lengths_replay_exactly(self):
        for n in (4000,8001,28827,28847,160000):
            with self.subTest(frames=n):
                clean=tone(n); saved=clean.copy()
                first,details=mp3_roundtrip(clean)
                again,repeated=mp3_roundtrip(clean)
                self.assertEqual(len(first),n)
                self.assertEqual(first.dtype,np.float32)
                self.assertEqual(details,repeated)
                np.testing.assert_array_equal(first,again)
                np.testing.assert_array_equal(clean,saved)
                self.assertEqual(details["gapless_codec_frames"],n+(-n)%576)
                self.assertEqual(details["protocol_zero_tail_samples"],(-n)%576)
                self.assertFalse(details["alignment_search_performed"])
                self.assertGreater(float(np.mean((first-clean)**2)),0)

    @unittest.skipUnless(HAS_CODEC,"Install the optional compression group")
    def test_impulse_alignment_and_above_unit_float_output(self):
        impulse=np.zeros(8001,dtype=np.float32);impulse[2000]=.5
        output,_=mp3_roundtrip(impulse)
        self.assertLessEqual(abs(int(np.argmax(np.abs(output)))-2000),1)
        output,details=mp3_roundtrip(tone(amplitude=1.26))
        self.assertAlmostEqual(details["headroom_input_peak"],.63,places=5)
        self.assertGreater(float(np.max(np.abs(output))),1)
        self.assertFalse(details["clipping_applied"])
        self.assertEqual(len(codec_identity()["binary_sha256"]),64)

    @unittest.skipUnless(HAS_CODEC,"Install the optional compression group")
    def test_real_runner_preserves_partitions_and_only_retains_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);arguments,rows=fixtures.AuditTests().fixture(root)
            audit_recordings(**arguments)
            kwargs=dict(manifest=arguments["manifest"],audio_root=arguments["audio_root"],
                        recording_audit=root/"audit/report.json",recordings=root/"audit/recordings.json",output_dir=root/"compression")
            report=audit_compression(**kwargs)
            self.assertEqual(report["gapless_sample_counts_verified"],4)
            self.assertEqual(report["identical_bitstream_and_sample_replays"],4)
            self.assertFalse(report["evaluation_audio_processed"])
            self.assertFalse(report["models_refitted"])
            self.assertFalse(report["thresholds_changed"])
            self.assertEqual({p.name for p in (root/"compression").iterdir()},{"report.json","recordings.json"})
            with self.assertRaisesRegex(ValueError,"already exists"):
                audit_compression(**kwargs)
            rows.append(dict(audio_path="eval/unopened.flac",label="bonafide",split="eval",speaker_id="other",attack_id="-"))
            fixtures.AuditTests().write_manifest(arguments["manifest"],rows)
            with patch("voicefake.compression.load_audio") as loader:
                with self.assertRaisesRegex(ValueError,"evaluation audio"):
                    audit_compression(**{**kwargs,"output_dir":root/"forbidden"})
                loader.assert_not_called()


if __name__=="__main__":
    unittest.main()
