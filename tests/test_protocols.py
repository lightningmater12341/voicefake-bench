"""Generated metadata fixtures exercise import and sampling; no audio results."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from voicefake.protocols import build_manifest, parse_protocol, select_subset


def fixture(split, speaker):
    prefix = {"train": "T", "dev": "D", "eval": "E"}[split]
    attack = "A07" if split == "eval" else "A01"
    return "\n".join(
        f"{speaker} LA_{prefix}_{i:07d} - {('-' if i < 3 else attack)} "
        f"{('bonafide' if i < 3 else 'spoof')}" for i in range(1, 6)
    ) + "\n"


class ProtocolTests(unittest.TestCase):
    def test_real_format_and_path(self):
        row = parse_protocol("LA_0001 LA_T_0000001 - - bonafide\n", "train")[0]
        self.assertEqual(row["audio_path"], "ASVspoof2019_LA_train/flac/LA_T_0000001.flac")
        self.assertEqual(row["speaker_id"], "LA_0001")

    def test_wrong_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "official split"):
            parse_protocol(fixture("train", "LA_0001"), "dev")

    def test_duplicate_recording_rejected(self):
        line = "LA_0001 LA_T_0000001 - - bonafide\n"
        with self.assertRaisesRegex(ValueError, "duplicate recording"):
            parse_protocol(line + line, "train")

    def test_protocol_errors(self):
        invalid = ["LA_0001 LA_T_0000001 - bonafide",
                   "LA_0001 LA_T_0000001 x - bonafide",
                   "LA_0001 LA_T_0000001 - - fake",
                   "LA_0001 LA_T_0000001 - A07 spoof",
                   "LA_0001 LA_T_0000001 - A01 bonafide"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_protocol(text, "train")

    def test_empty_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError, "no records"):
            parse_protocol("\n", "train")

    def test_sampling_order_independence_and_nested_caps(self):
        rows = parse_protocol(fixture("train", "LA_0001"), "train")
        small = select_subset(rows, 1, 42)
        self.assertEqual(small, select_subset(list(reversed(rows)), 1, 42))
        self.assertTrue({r["audio_path"] for r in small}.issubset(
            {r["audio_path"] for r in select_subset(rows, 2, 42)}))
        self.assertEqual(len(small), 2)
        self.assertEqual(len(select_subset(rows, None)), 5)

    def test_invalid_cap_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            select_subset([], 0)

    def inputs(self, root, overlap=False):
        paths = {}
        for i, split in enumerate(("train", "dev", "eval"), 1):
            path = root / f"{split}.txt"
            path.write_text(fixture(split, "LA_0001" if overlap else f"LA_{i:04d}"))
            paths[split] = path
        return paths

    def test_import_hashes_counts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.inputs(root)
            output = root / "subset.csv"
            report = build_manifest(paths, output, per_stratum=1)
            self.assertEqual(report["full_input_audit"]["rows"], 15)
            self.assertEqual(report["selected_audit"]["rows"], 6)
            self.assertEqual(report["manifest_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output.with_suffix(".csv.json").read_text()), report)
            with output.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            with self.assertRaisesRegex(ValueError, "already exists"):
                build_manifest(paths, output)

    def test_full_source_leakage_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.inputs(root, overlap=True)
            output = root / "subset.csv"
            with self.assertRaisesRegex(ValueError, "overlaps"):
                build_manifest(paths, output, per_stratum=1)
            self.assertFalse(output.exists())

    def test_three_protocols_required(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            build_manifest({}, "unused.csv")


if __name__ == "__main__":
    unittest.main()
