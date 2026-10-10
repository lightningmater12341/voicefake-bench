"""Import official ASVspoof 2019 LA CM protocols without reading audio."""

import csv
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from .manifest import REQUIRED_COLUMNS, SPLITS, validate_manifest

SPLIT_PREFIX = {"train": "T", "dev": "D", "eval": "E"}
SPLIT_ORDER = {"train": 0, "dev": 1, "eval": 2}


def parse_protocol(text: str, split: str) -> list[dict[str, str]]:
    """Parse the five-field countermeasure key, retaining official membership."""
    if split not in SPLITS:
        raise ValueError(f"Invalid protocol split {split!r}")
    rows = []
    clip_ids = set()
    for number, line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{split} line {number}: expected five CM protocol fields")
        speaker, clip, placeholder, attack, label = fields
        if not re.fullmatch(r"LA_\d{4}", speaker):
            raise ValueError(f"{split} line {number}: invalid LA speaker ID")
        if not re.fullmatch(rf"LA_{SPLIT_PREFIX[split]}_\d{{7}}", clip):
            raise ValueError(f"{split} line {number}: clip ID does not match its official split")
        if placeholder != "-":
            raise ValueError(f"{split} line {number}: expected '-' in CM field three")
        if label not in {"bonafide", "spoof"}:
            raise ValueError(f"{split} line {number}: invalid label {label!r}")
        allowed_attacks = range(1, 7) if split != "eval" else range(7, 20)
        if label == "bonafide" and attack != "-":
            raise ValueError(f"{split} line {number}: bonafide attack ID must be '-'")
        if label == "spoof" and attack not in {f"A{a:02d}" for a in allowed_attacks}:
            raise ValueError(f"{split} line {number}: unexpected spoof attack ID {attack!r}")
        if clip in clip_ids:
            raise ValueError(f"{split} line {number}: duplicate recording ID {clip!r}")
        clip_ids.add(clip)
        rows.append({"audio_path": f"ASVspoof2019_LA_{split}/flac/{clip}.flac",
                     "label": label, "split": split, "speaker_id": speaker,
                     "attack_id": attack})
    if not rows:
        raise ValueError(f"{split}: protocol has no records")
    return rows


def select_subset(rows: list[dict[str, str]], per_stratum: int | None,
                  seed: int = 42) -> list[dict[str, str]]:
    """Select a fixed cap per (split, label, attack); independent of input order."""
    if per_stratum is not None and (not isinstance(per_stratum, int) or per_stratum < 1):
        raise ValueError("per_stratum must be a positive integer or None for all records")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["label"], row["attack_id"])].append(row)
    selected = []
    for key in sorted(groups):
        def rank(row):
            identity = json.dumps([seed, *key, row["audio_path"]], separators=(",", ":"))
            return (hashlib.sha256(identity.encode("utf-8")).hexdigest(), row["audio_path"])
        ordered = sorted(groups[key], key=rank)
        selected.extend(ordered if per_stratum is None else ordered[:per_stratum])
    return sorted(selected, key=lambda r: (SPLIT_ORDER[r["split"]], r["audio_path"]))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(protocols: dict[str, str | Path], output: str | Path,
                   per_stratum: int | None = 100, seed: int = 42) -> dict:
    """Audit full input protocols, then write a manifest and provenance sidecar.

    Inputs must contain all three official CM keys. No audio is downloaded,
    scored, or checked for existence. Existing output files are never replaced.
    """
    if set(protocols) != SPLITS:
        raise ValueError("Provide exactly train, dev, and eval CM protocol files")
    target = Path(output)
    sidecar = target.with_suffix(target.suffix + ".json")
    if target.exists() or sidecar.exists():
        raise ValueError("Output or provenance sidecar already exists; choose a new output path")
    rows = []
    sources = {}
    for split in ("train", "dev", "eval"):
        path = Path(protocols[split])
        body = path.read_bytes()
        parsed = parse_protocol(body.decode("utf-8-sig"), split)
        rows.extend(parsed)
        sources[split] = {"filename": path.name, "sha256": hashlib.sha256(body).hexdigest(),
                          "rows": len(parsed)}
    selected = select_subset(rows, per_stratum, seed)
    # Validate the complete inputs before sampling, so a cap cannot hide leakage.
    with tempfile.TemporaryDirectory() as folder:
        complete = Path(folder) / "complete.csv"
        subset = Path(folder) / "subset.csv"
        _write_csv(complete, rows)
        full_audit = validate_manifest(complete)
        _write_csv(subset, selected)
        selected_audit = validate_manifest(subset)
        manifest_bytes = subset.read_bytes()
    groups = defaultdict(lambda: {"available": 0, "selected": 0})
    for row in rows:
        groups[(row["split"], row["label"], row["attack_id"])]["available"] += 1
    for row in selected:
        groups[(row["split"], row["label"], row["attack_id"])]["selected"] += 1
    report = {
        "schema_version": 1, "dataset": "ASVspoof 2019 LA",
        "dataset_doi": "https://doi.org/10.7488/ds/2555",
        "dataset_license": "https://opendatacommons.org/licenses/by/1.0/",
        "selection": {"algorithm": "sha256-ranking-v1", "seed": seed,
                      "per_split_label_attack_cap": per_stratum,
                      "mode": "full" if per_stratum is None else "subset"},
        "sources": sources, "full_input_audit": full_audit, "selected_audit": selected_audit,
        "strata": [{"split": key[0], "label": key[1], "attack_id": key[2], **groups[key]}
                   for key in sorted(groups)],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "audio_files_checked": False, "model_results": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(manifest_bytes)
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
