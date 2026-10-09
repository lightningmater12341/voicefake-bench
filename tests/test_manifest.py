"""Small metadata fixtures test leakage protection, not model performance."""

import tempfile
import unittest
from pathlib import Path

from voicefake.manifest import validate_manifest


HEADER = "audio_path,label,split,speaker_id,attack_id\n"


class ManifestTests(unittest.TestCase):
    def audit(self, rows, header=HEADER):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            path.write_text(header + rows, encoding="utf-8")
            return validate_manifest(path)

    def test_valid_manifest_counts(self):
        result = self.audit(
            "train/a.flac,bonafide,train,s1,-\n"
            "train/b.flac,spoof,train,s1,A01\n"
            "dev/c.flac,spoof,dev,s2,A01\n"
        )
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["speakers"], 2)
        self.assertEqual(result["counts"]["train"], {"bonafide": 1, "spoof": 1})

    def test_speaker_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.audit("a.flac,bonafide,train,s1,-\nb.flac,spoof,dev,s1,A01\n")

    def test_duplicate_path_alias_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate audio_path"):
            self.audit("train/a.flac,bonafide,train,s1,-\ntrain\\A.flac,spoof,train,s1,A01\n")

    def test_missing_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.audit("a.flac,spoof,train,,A01\n")

    def test_invalid_label_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid label"):
            self.audit("a.flac,fake,train,s1,A01\n")

    def test_label_attack_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "attack_id"):
            self.audit("a.flac,bonafide,train,s1,A01\n")

    def test_root_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, "root-relative"):
            self.audit("../a.flac,spoof,train,s1,A01\n")

    def test_empty_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, "no data rows"):
            self.audit("")


if __name__ == "__main__":
    unittest.main()
