"""Fixed per-recording perturbations applied after the existing clean audio recipe."""

import csv
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .audio import AudioConfig, _file_hash, load_audio
from .audit import waveform_digest
from .feature_cache import runtime_identity
from .manifest import REQUIRED_COLUMNS, validate_manifest


@dataclass(frozen=True)
class PerturbationConfig:
    schema_version: int = 1
    seed: int = 42
    sample_rate: int = 16000
    noise_snrs_db: tuple = (20, 10)
    bandwidth_rate: int = 8000

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported perturbation schema")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(self.sample_rate) is not int or self.sample_rate != 16000 or type(self.bandwidth_rate) is not int or self.bandwidth_rate != 8000:
            raise ValueError("Version 1 uses exactly a 16000 -> 8000 -> 16000 Hz round trip")
        if not isinstance(self.noise_snrs_db, (tuple, list)) or not 1 <= len(self.noise_snrs_db) <= 4:
            raise ValueError("Provide one to four noise SNRs")
        for value in self.noise_snrs_db:
            if type(value) is not int or not 0 <= value <= 60:
                raise ValueError("Noise SNR must be an integer in [0, 60] dB")
        if len(set(self.noise_snrs_db)) != len(self.noise_snrs_db):
            raise ValueError("Noise SNRs must be distinct")
        object.__setattr__(self, "noise_snrs_db", tuple(self.noise_snrs_db))

    @property
    def conditions(self):
        return tuple(f"noise_{value:g}db" for value in self.noise_snrs_db) + ("bandwidth_8khz",)

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Perturbation configuration must be a JSON object")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown perturbation configuration fields") from exc


def apply_perturbation(waveform: np.ndarray, condition: str,
                       config: PerturbationConfig = PerturbationConfig()):
    """Labels, filenames and traversal order are deliberately absent from this API."""
    clean = np.asarray(waveform)
    if clean.ndim != 1 or clean.dtype != np.float32 or len(clean) < 4 or not np.isfinite(clean).all() or not np.any(clean):
        raise ValueError("Require finite, nonzero, mono float32 clean audio with at least four samples")
    if condition not in config.conditions:
        raise ValueError("Unknown perturbation condition")
    identity = waveform_digest(clean, config.sample_rate, "<f4")
    signal = clean.astype(np.float64)
    power = float(np.mean(signal ** 2))
    noise_levels = dict(zip(config.conditions[:-1], config.noise_snrs_db))
    details = {"condition": condition, "clean_waveform_sha256": identity}
    if condition in noise_levels:
        snr = float(noise_levels[condition])
        # Exclude SNR from this seed so both levels use the same noise direction, at different scales.
        seed_body = json.dumps(["voicefake-noise-v1", config.seed, identity], separators=(",", ":")).encode()
        seed_hash = hashlib.sha256(seed_body).hexdigest()
        generator = np.random.Generator(np.random.PCG64(int(seed_hash, 16)))
        noise = generator.standard_normal(len(clean), dtype=np.float64)
        noise *= math.sqrt(power / (10 ** (snr / 10) * float(np.mean(noise ** 2))))
        output = (signal + noise).astype(np.float32)
        actual_noise_power = float(np.mean((output.astype(np.float64) - signal) ** 2))
        if actual_noise_power <= 0:
            raise ValueError("Float32 cannot represent the requested noise condition")
        measured = 10 * math.log10(power / actual_noise_power)
        if abs(measured - snr) > 0.001:
            raise ValueError("Measured post-cast SNR differs from the target by more than 0.001 dB")
        details.update(target_snr_db=snr, measured_snr_db=measured, noise_seed_sha256=seed_hash,
                       rng="numpy.Generator(PCG64); float64 standard_normal; empirical RMS scaling")
    else:
        narrow = resample_poly(signal, 1, 2, window=("kaiser", 5.0), padtype="constant", cval=0.0)
        restored = resample_poly(narrow, 2, 1, window=("kaiser", 5.0), padtype="constant", cval=0.0)
        output = restored[:len(clean)].astype(np.float32)
        details.update(intermediate_sample_rate=config.bandwidth_rate, intermediate_frames=len(narrow),
                       right_tail_samples_removed=len(restored) - len(clean),
                       filter="polyphase low-pass FIR; Kaiser beta=5.0; constant-zero boundaries; float64 both passes")
    output = np.ascontiguousarray(output, dtype=np.float32)
    if output.shape != clean.shape or not np.isfinite(output).all():
        raise ValueError("Perturbation changed frame count or produced non-finite samples")
    details.update(sample_rate=config.sample_rate, output_frames=len(output), output_dtype="float32",
                   output_sha256=waveform_digest(output, config.sample_rate, "<f4"),
                   absolute_peak=float(np.max(np.abs(output))), above_unit_samples=int(np.sum(np.abs(output) > 1)),
                   output_input_rms_ratio=float(np.sqrt(np.mean(output.astype(np.float64) ** 2) / power)),
                   clipping_applied=False, gain_normalization_applied=False)
    return output, details


def audit_perturbations(manifest: str | Path, audio_root: str | Path, audit_report: str | Path,
                       recordings: str | Path, output_dir: str | Path,
                       config: PerturbationConfig = PerturbationConfig()) -> dict:
    target = Path(output_dir)
    if target.exists():
        raise ValueError("Perturbation output already exists; choose a new run directory")
    metadata = validate_manifest(manifest)
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        rows = [{name: row[name].strip() for name in REQUIRED_COLUMNS} for row in csv.DictReader(stream)]
    if {row["split"] for row in rows} != {"train", "dev"}:
        raise ValueError("Perturbation audit requires train/dev only; evaluation audio is unavailable before freeze")
    source = json.loads(Path(audit_report).read_text(encoding="utf-8"))
    if source["status"] != "pass" or source["evaluation_audio_processed"] is not False:
        raise ValueError("Require a clean prior recording audit")
    if source["source_manifest_sha256"] != _file_hash(Path(manifest)) or source["manifest_audit"] != metadata:
        raise ValueError("Manifest differs from the prior recording audit")
    if source["local_artifact_sha256"]["recordings.json"] != _file_hash(Path(recordings)):
        raise ValueError("Prior recording evidence checksum differs")
    evidence = json.loads(Path(recordings).read_text(encoding="utf-8"))
    if [{name: row[name] for name in REQUIRED_COLUMNS} for row in evidence] != rows or source["recordings_checked"] != len(rows):
        raise ValueError("Prior recording membership differs")
    if source["runtime"] != runtime_identity() or source["audit_source_sha256"]["audio.py"] != _file_hash(Path(__file__).with_name("audio.py")):
        raise ValueError("Clean audio runtime or loader code differs from prior evidence")
    audio_config = AudioConfig(**source["audio_config"])
    if audio_config.target_sample_rate != config.sample_rate:
        raise ValueError("Perturbation rate differs from the clean audio recipe")
    root = Path(audio_root).resolve(strict=True)
    paths = [(root / row["audio_path"]).resolve(strict=True) for row in rows]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Resolved audio path escapes the dataset root")
    details = []
    for row, prior, path in zip(rows, evidence, paths):
        clean, loaded = load_audio(path, audio_config)
        identity = waveform_digest(clean, config.sample_rate, "<f4")
        if loaded["file_sha256"] != prior["encoded_sha256"] or identity != prior["processed_sha256"]:
            raise ValueError("Clean source no longer matches the audited model input")
        transformed = {}
        for condition in config.conditions:
            output, first = apply_perturbation(clean, condition, config)
            replay, second = apply_perturbation(clean, condition, config)
            if first != second or not np.array_equal(output, replay):
                raise ValueError("Perturbation replay differed")
            transformed[condition] = first
        if waveform_digest(clean, config.sample_rate, "<f4") != identity:
            raise ValueError("A perturbation modified its clean source")
        details.append({**row, "conditions": transformed})
    summaries = {}
    for condition in config.conditions:
        values = [record["conditions"][condition] for record in details]
        summary = {"recordings": len(values), "shape_dtype_and_finite_checks_passed": len(values),
                   "identical_replays": len(values), "above_unit_samples": sum(v["above_unit_samples"] for v in values),
                   "recordings_above_unit_peak": sum(v["absolute_peak"] > 1 for v in values),
                   "maximum_absolute_peak": max(v["absolute_peak"] for v in values)}
        if "measured_snr_db" in values[0]:
            summary.update(minimum_measured_snr_db=min(v["measured_snr_db"] for v in values),
                           maximum_measured_snr_db=max(v["measured_snr_db"] for v in values))
        else:
            summary["odd_length_tail_trims"] = sum(v["right_tail_samples_removed"] for v in values)
        summaries[condition] = summary
    report = {"schema_version": 1, "scope": "audio-property audit on the existing ASVspoof 2019 LA train/dev subset; no detector results",
              "config": {**asdict(config), "noise_snrs_db": list(config.noise_snrs_db)}, "conditions": summaries, "recordings_checked": len(rows),
              "transforms_checked": len(rows) * len(config.conditions), "manifest_audit": metadata,
              "source_manifest_sha256": _file_hash(Path(manifest)),
              "source_recording_audit_sha256": _file_hash(Path(audit_report)),
              "source_recording_evidence_sha256": _file_hash(Path(recordings)),
              "source_feature_index_sha256": source["source_feature_index_sha256"],
              "clean_audio_config": source["audio_config"], "runtime": runtime_identity(),
              "source_code_sha256": {name: _file_hash(Path(__file__).with_name(name))
                                     for name in ("perturbations.py", "audio.py", "audit.py")},
              "partition_membership_preserved": True, "clean_sources_unchanged": True,
              "evaluation_audio_processed": False, "models_refitted": False, "thresholds_changed": False,
              "detector_scores_computed": False, "training_augmentation_performed": False,
              "waveform_files_written": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-perturb-", dir=target.parent) as folder:
        staged = Path(folder)
        (staged / "transforms.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["local_artifact_sha256"] = {"transforms.json": _file_hash(staged / "transforms.json")}
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(target)
    return report
