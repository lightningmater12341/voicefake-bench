"""Validate dataset metadata before any audio enters a model."""

import csv
from collections import Counter
from pathlib import Path, PurePosixPath


REQUIRED_COLUMNS = ("audio_path", "label", "split", "speaker_id", "attack_id")
LABELS = {"bonafide", "spoof"}
SPLITS = {"train", "dev", "eval"}


def validate_manifest(path: str | Path) -> dict:
    """Return metadata counts; reject malformed rows and partition leakage.

    Paths are dataset-root-relative identifiers, not filesystem access. Checking
    file contents and duplicate audio hashes belongs to the later audio audit.
    A partial manifest is allowed so import and development can be incremental.
    """
    counts = Counter()
    paths = set()
    speaker_splits = {}
    attacks = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names in manifest header")
        missing = set(REQUIRED_COLUMNS) - set(header)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Line {line_number}: too many CSV fields")
            values = {name: (row.get(name) or "").strip() for name in REQUIRED_COLUMNS}
            if not all(values.values()):
                raise ValueError(f"Line {line_number}: required fields must be nonempty")
            label, split = values["label"], values["split"]
            if label not in LABELS:
                raise ValueError(f"Line {line_number}: invalid label {label!r}")
            if split not in SPLITS:
                raise ValueError(f"Line {line_number}: invalid split {split!r}")
            raw_path = values["audio_path"].replace("\\", "/")
            audio_path = PurePosixPath(raw_path)
            if audio_path.is_absolute() or ".." in audio_path.parts or ":" in raw_path:
                raise ValueError(f"Line {line_number}: audio_path must be root-relative")
            if str(audio_path) == ".":
                raise ValueError(f"Line {line_number}: audio_path must name a file")
            path_key = str(audio_path).casefold()
            if path_key in paths:
                raise ValueError(f"Line {line_number}: duplicate audio_path {raw_path!r}")
            paths.add(path_key)
            speaker = values["speaker_id"]
            previous_split = speaker_splits.setdefault(speaker, split)
            if previous_split != split:
                raise ValueError(
                    f"Line {line_number}: speaker {speaker!r} overlaps "
                    f"{previous_split!r} and {split!r}"
                )
            attack = values["attack_id"]
            if (label == "bonafide") != (attack == "-"):
                raise ValueError(
                    f"Line {line_number}: bonafide requires attack_id '-', "
                    "spoof requires an attack identifier"
                )
            if label == "spoof":
                attacks.add(attack)
            counts[(split, label)] += 1
    if not counts:
        raise ValueError("Manifest has no data rows")
    return {
        "rows": sum(counts.values()),
        "speakers": len(speaker_splits),
        "spoofing_methods": sorted(attacks),
        "counts": {
            split: {label: counts[(split, label)] for label in sorted(LABELS)}
            for split in sorted(SPLITS)
        },
    }
