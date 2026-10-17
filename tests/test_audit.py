"""Synthetic recordings test exact-content leakage, never detector performance."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voicefake.audio import AudioConfig, _file_hash
from voicefake.audit import audit_recordings, duplicate_groups, fingerprint_recording, waveform_digest
from voicefake.feature_cache import extract_manifest
from voicefake.manifest import REQUIRED_COLUMNS
from voicefake.protocols import parse_protocol


def signal(frequency=440):
    raw = .2 * np.sin(2 * np.pi * frequency * np.arange(8000) / 16000)
    return (np.rint(raw * 32768) / 32768).astype(np.float32)


class AuditTests(unittest.TestCase):
    def write_manifest(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader(); writer.writerows(rows)

    def fixture(self, root, duplicate=None):
        keys = {
            "train": "LA_1001 LA_T_1000001 - - bonafide\nLA_1001 LA_T_1000002 - A01 spoof\n",
            "dev": "LA_2001 LA_D_2000001 - - bonafide\nLA_2001 LA_D_2000002 - A01 spoof\n",
            "eval": "LA_3001 LA_E_3000001 - - bonafide\n",
        }
        protocols, rows, reference = {}, [], {"sources": {}}
        for split, body in keys.items():
            path = root / (split + ".txt"); path.write_text(body, encoding="utf-8")
            protocols[split] = path; reference["sources"][split] = {"sha256": _file_hash(path)}
            if split != "eval":
                rows.extend(parse_protocol(body, split))
        audio_root = root / "audio"; audio_root.mkdir()
        for i, row in enumerate(rows):
            path = audio_root / row["audio_path"]; path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, signal(300 + i * 111), 16000, subtype="PCM_16")
        if duplicate:
            destination = rows[3] if duplicate == "label_conflict" else rows[2]
            shutil.copyfile(audio_root / rows[0]["audio_path"], audio_root / destination["audio_path"])
        manifest = root / "manifest.csv"; self.write_manifest(manifest, rows)
        reference_path = root / "pinned.json"; reference_path.write_text(json.dumps(reference), encoding="utf-8")
        index = root / "index.json"
        extract_manifest(manifest, audio_root, root / "cache", index)
        return dict(manifest=manifest, audio_root=audio_root, protocols=protocols, protocol_reference=reference_path,
                    index_path=index, cache_dir=root / "cache", output_dir=root / "audit"), rows

    def test_container_change_preserves_decoded_and_processed_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sf.write(root / "one.flac", signal(), 16000, subtype="PCM_16")
            sf.write(root / "two.wav", signal(), 16000, subtype="PCM_16")
            first, second = [fingerprint_recording(root / name) for name in ("one.flac", "two.wav")]
            self.assertNotEqual(first["encoded_sha256"], second["encoded_sha256"])
            self.assertEqual(first["decoded_sha256"], second["decoded_sha256"])
            self.assertEqual(first["processed_sha256"], second["processed_sha256"])

    def test_different_channels_can_collapse_after_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); mono = signal()
            sf.write(root / "mono.wav", mono, 16000, subtype="PCM_16")
            sf.write(root / "stereo.wav", np.column_stack([mono + .03125, mono - .03125]), 16000, subtype="PCM_16")
            first, second = [fingerprint_recording(root / name) for name in ("mono.wav", "stereo.wav")]
            self.assertNotEqual(first["decoded_sha256"], second["decoded_sha256"])
            self.assertEqual(first["processed_sha256"], second["processed_sha256"])

    def test_hash_includes_shape_rate_dtype_and_canonical_signed_zero(self):
        values = np.array([0., -0., .5, -.5])
        positive_zero = values.copy(); positive_zero[positive_zero == 0] = 0
        digest = waveform_digest(values, 16000, "<f8")
        self.assertEqual(digest, waveform_digest(positive_zero.astype(">f8"), 16000, "<f8"))
        self.assertNotEqual(digest, waveform_digest(values.reshape(2, 2), 16000, "<f8"))
        self.assertNotEqual(digest, waveform_digest(values, 8000, "<f8"))
        self.assertNotEqual(digest, waveform_digest(values, 16000, "<f4"))

    def test_duration_guard_precedes_full_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "long.wav"; sf.write(path, np.tile(signal(), 4), 16000)
            with patch.object(sf.SoundFile, "read", side_effect=AssertionError("Unexpected decode")):
                with self.assertRaisesRegex(ValueError, "refusing full decode"):
                    fingerprint_recording(path, AudioConfig(max_duration_seconds=1, max_source_duration_seconds=1))

    def test_clean_audit_replays_and_does_not_read_evaluation_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args, _ = self.fixture(root)
            first = audit_recordings(**args)
            second = audit_recordings(**{**args, "output_dir": root / "replay"})
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "pass")
            self.assertEqual(first["official_rows_matched"], 4)
            self.assertEqual(first["full_source_metadata_audit"]["rows"], 5)
            self.assertEqual(first["feature_sources_and_waveforms_verified"], 4)
            self.assertFalse(first["evaluation_audio_processed"])
            self.assertFalse(first["models_refitted"])
            self.assertFalse(first["thresholds_changed"])
            # There is deliberately no evaluation audio file in the fixture.
            for name, digest in first["local_artifact_sha256"].items():
                self.assertEqual(_file_hash(root / "audit" / name), digest)
            with self.assertRaisesRegex(ValueError, "already exists"):
                audit_recordings(**args)

    def test_duplicate_and_contradictory_labels_are_preserved_for_review(self):
        for mode in ("encoded", "label_conflict"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); args, _ = self.fixture(root, mode)
                report = audit_recordings(**args)
                self.assertEqual(report["status"], "review_required")
                self.assertTrue(report["cross_split_content_detected"])
                self.assertEqual(report["recordings_removed"], 0)
                self.assertEqual(report["contradictory_labels_detected"], mode == "label_conflict")
                self.assertEqual(report["duplicate_summary"]["encoded_sha256"]["groups"], 1)
                details = json.loads((root / "audit/duplicate_groups.json").read_text(encoding="utf-8"))
                self.assertEqual(len(details["decoded_sha256"][0]["members"]), 2)

    def test_within_partition_duplicate_is_reported(self):
        rows = [{"audio_path": str(i), "label": "bonafide", "split": "train", "speaker_id": "s",
                 "attack_id": "-", "hash": "same"} for i in range(3)]
        summary, _ = duplicate_groups(rows, "hash")
        self.assertEqual(summary["groups"], 1)
        self.assertEqual(summary["redundant_rows"], 2)
        self.assertEqual(summary["cross_split_groups"], 0)
        self.assertEqual(summary["within_split_groups"], 1)

    def test_protocol_pin_and_selected_membership_checked_before_audio(self):
        for mode in ("pin", "membership"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); args, rows = self.fixture(root)
                if mode == "pin":
                    path = args["protocols"]["train"]
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                else:
                    rows[0].update(label="spoof", attack_id="A01")
                    self.write_manifest(args["manifest"], rows)
                with patch("voicefake.audit.fingerprint_recording") as decoder:
                    with self.assertRaisesRegex(ValueError, "pinned"):
                        audit_recordings(**args)
                    decoder.assert_not_called()

    def test_full_keys_catch_speaker_overlap_outside_selected_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args, _ = self.fixture(root)
            path = args["protocols"]["eval"]
            path.write_text("LA_1001 LA_E_3000001 - - bonafide\n", encoding="utf-8")
            reference = json.loads(args["protocol_reference"].read_text(encoding="utf-8"))
            reference["sources"]["eval"]["sha256"] = _file_hash(path)
            args["protocol_reference"].write_text(json.dumps(reference), encoding="utf-8")
            with patch("voicefake.audit.fingerprint_recording") as decoder:
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    audit_recordings(**args)
                decoder.assert_not_called()

    def test_evaluation_audio_membership_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args, rows = self.fixture(root)
            rows.extend(parse_protocol(args["protocols"]["eval"].read_text(encoding="utf-8"), "eval"))
            self.write_manifest(args["manifest"], rows)
            with patch("voicefake.audit.fingerprint_recording") as decoder:
                with self.assertRaisesRegex(ValueError, "evaluation audio"):
                    audit_recordings(**args)
                decoder.assert_not_called()

    def test_changed_audio_cannot_reuse_existing_model_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args, rows = self.fixture(root)
            sf.write(args["audio_root"] / rows[0]["audio_path"], signal(1250), 16000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "Recording bytes differ"):
                audit_recordings(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_changed_cached_feature_is_rejected_before_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args, _ = self.fixture(root)
            index = json.loads(args["index_path"].read_text(encoding="utf-8"))
            np.save(args["cache_dir"] / index["records"][0]["feature_file"], np.zeros(40, dtype=np.float32))
            with patch("voicefake.audit.fingerprint_recording") as decoder:
                with self.assertRaisesRegex(ValueError, "Feature file checksum"):
                    audit_recordings(**args)
                decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
