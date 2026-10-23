"""Read-only local inference with the frozen primary model; never a new evaluation."""

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .audio import AudioConfig, _file_hash, load_audio
from .audit import waveform_digest
from .comparison import model_scores
from .features import FeatureConfig, extract_features
from .freeze import _inside, verify_freeze
from .metrics import threshold_predictions


def _evaluation_guard(root, specification, lock):
    """Use completed audit fingerprints without opening any evaluation audio."""
    audit = json.loads((root / "docs/EVALUATION_CONTENT_AUDIT.json").read_text(encoding="utf-8"))
    manifest = _inside(root, specification["dataset"]["evaluation_manifest"])
    if (audit["status"] != "pass" or audit["frozen_lock_sha256"] != lock
            or audit["source_eval_manifest_sha256"] != _file_hash(manifest)):
        raise ValueError("Completed evaluation audit does not match the frozen experiment")
    fingerprints = root / "artifacts/evaluation-audit/day13/recordings.json"
    if _file_hash(fingerprints) != audit["local_artifact_sha256"]["recordings.json"]:
        raise ValueError("Evaluation guard fingerprints changed")
    records = json.loads(fingerprints.read_text(encoding="utf-8"))
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = ("audio_path", "label", "split", "speaker_id", "attack_id")
    if (not rows or len(records) != len(rows)
            or any(r["split"] != "eval" for r in rows)
            or [{k: r[k] for k in fields} for r in records] != rows):
        raise ValueError("Evaluation guard membership differs")
    paths = {(root / "data/asvspoof2019/LA" / r["audio_path"]).resolve() for r in rows}
    return paths, {r["encoded_sha256"] for r in records}, {r["processed_sha256"] for r in records}


def demo_audio(path: str | Path, repo_root: str | Path = ".") -> dict:
    root = Path(repo_root).resolve(strict=True)
    specification_path = root / "configs/frozen-experiment.json"
    verified = verify_freeze(specification_path, root)
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    primary = specification["primary_model"]
    if primary != "svm":
        raise ValueError("This demo requires the original frozen primary SVM")
    details = specification["models"][primary]
    paths, encoded, processed = _evaluation_guard(root, specification, verified["lock_sha256"])
    source = Path(path).resolve(strict=True)
    if source in paths or _file_hash(source) in encoded:
        raise ValueError("Held-out evaluation recording refused; use development or your own audio")
    config = AudioConfig.from_json(root / "configs/audio.json")
    feature_config = FeatureConfig.from_json(root / "configs/features.json")
    waveform, audio = load_audio(source, config)
    # Check the bytes actually decoded too, in case the input changed after the first hash.
    if (audio["file_sha256"] in encoded
            or waveform_digest(waveform, config.target_sample_rate, "<f4") in processed):
        raise ValueError("Exact evaluation content refused before model inference")
    with threadpool_limits(limits=1):
        vector, _ = extract_features(waveform, config.target_sample_rate, feature_config)
        # Only the repository-relative, freeze-verified artifact may be deserialized.
        model = joblib.load(_inside(root, details["artifact"]))
        scores = model_scores(model, vector.reshape(1, -1).astype(np.float64), "margin")
        flagged = bool(threshold_predictions(scores, details["decision_rule"])[0])
    return {
        "schema_version": 1,
        "scope": "experimental local single-recording demo; not benchmark evidence",
        "decision": "flagged_as_spoof" if flagged else "not_flagged_as_spoof",
        "svm_margin": float(scores[0]),
        "decision_rule": details["decision_rule"],
        "score_is_probability": False,
        "interpretation": "A flag is a model decision, not proof of origin; no flag does not verify genuine speech.",
        "limitation": "Older ASVspoof 2019 subset; noise, bandwidth and MP3 can cause large false-alarm shifts.",
        "primary_model": primary,
        "frozen_lock_sha256": verified["lock_sha256"],
        "model_sha256": details["sha256"],
        "feature_count": int(vector.size),
        "feature_float32_le_sha256": hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest(),
        "input": {key: value for key, value in audio.items() if key != "filename"},
        "evaluation_guard": "original paths, encoded hashes and exact processed-waveform hashes checked",
        "models_refitted": False,
        "threshold_changed": False,
        "files_written": False,
    }
