"""Auditable warm-process costs on frozen clean development data only."""

import csv
import gc
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .audio import AudioConfig, _file_hash, load_audio
from .audit import waveform_digest
from .comparison import model_scores
from .feature_cache import _vector_hash
from .features import FeatureConfig, extract_features
from .freeze import MODELS, _inside, verify_freeze
from .manifest import REQUIRED_COLUMNS, validate_manifest
from .metrics import threshold_predictions


@dataclass(frozen=True)
class CostConfig:
    schema_version: int = 1
    feature_repeats: int = 3
    single_clip_repeats: int = 3
    batch_repeats: int = 7
    warmup_clips: int = 5
    thread_limit: int = 1

    def __post_init__(self):
        expected = {"schema_version": 1, "feature_repeats": 3, "single_clip_repeats": 3,
                    "batch_repeats": 7, "warmup_clips": 5, "thread_limit": 1}
        if any(type(getattr(self, key)) is not int or getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("Version 1 fixes three feature/single passes, seven batch calls and five warm-up clips with one thread")

    @classmethod
    def from_json(cls, path):
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("Cost configuration must be an object")
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Unknown cost configuration fields") from exc


def hardware_identity():
    """Allowlisted hardware only: never include hostnames, usernames, paths or serials."""
    result = {"system": platform.system(), "release": platform.release(), "os_version": platform.version(),
              "machine": platform.machine(), "cpu_description": platform.processor(),
              "logical_cpus": os.cpu_count(), "physical_memory_bytes": None, "power_ac_online": None}
    if os.name == "nt":
        import ctypes
        import winreg
        from ctypes import wintypes
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                result["cpu_description"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (key, ctypes.c_ulonglong) for key in ("total_physical", "available_physical", "total_page", "available_page", "total_virtual", "available_virtual", "extended_virtual")]
        memory = MemoryStatus(); memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result["physical_memory_bytes"] = int(memory.total_physical)
        class PowerStatus(ctypes.Structure):
            _fields_ = [("ac", wintypes.BYTE), ("flag", wintypes.BYTE), ("percent", wintypes.BYTE),
                        ("reserved", wintypes.BYTE), ("life", wintypes.DWORD), ("full_life", wintypes.DWORD)]
        power = PowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) and power.ac in (0, 1):
            result["power_ac_online"] = bool(power.ac)
    return result


def measure_call(function, clock=None):
    """Time only the callable; output verification and log writes happen afterwards."""
    clock = time.perf_counter_ns if clock is None else clock
    started = clock()
    result = function()
    elapsed = clock() - started
    if type(elapsed) is not int or elapsed <= 0:
        raise ValueError("Expected a positive integer monotonic-clock duration")
    return result, elapsed


def summarize_timings(records):
    if not records:
        raise ValueError("No timing records")
    for row in records:
        if type(row["elapsed_ns"]) is not int or row["elapsed_ns"] <= 0 or type(row["rows"]) is not int or row["rows"] < 1:
            raise ValueError("Timing duration and row counts must be positive integers")
        if not np.isfinite(row["audio_seconds"]) or row["audio_seconds"] <= 0:
            raise ValueError("Require a positive processed audio duration")
    total_ns = sum(row["elapsed_ns"] for row in records)
    total_rows = sum(row["rows"] for row in records)
    ms = np.array([row["elapsed_ns"] / 1e6 for row in records])
    audio_seconds = sum(row["audio_seconds"] for row in records)
    return {"timed_calls": len(records), "processed_rows_with_repeats": total_rows,
            "timed_total_ns": total_ns, "call_latency_ms": {"minimum": float(ms.min()), "p50": float(np.quantile(ms, .5, method="linear")),
                                                            "p95": float(np.quantile(ms, .95, method="linear")), "maximum": float(ms.max())},
            "amortized_ms_per_clip": total_ns / 1e6 / total_rows,
            "clips_per_timed_second": total_rows / (total_ns / 1e9),
            "processed_audio_seconds_with_repeats": audio_seconds,
            "real_time_factor": (total_ns / 1e9) / audio_seconds}


def pass_summaries(records):
    keys = sorted({(r["stage"], r["model"], r["mode"], r["repeat"]) for r in records})
    return [{"stage": stage, "model": model, "mode": mode, "repeat": repeat,
             **summarize_timings([r for r in records if (r["stage"], r["model"], r["mode"], r["repeat"]) == (stage, model, mode, repeat)])}
            for stage, model, mode, repeat in keys]


def run_costs(repo_root, output_dir="artifacts/costs/day12", progress=None):
    root = Path(repo_root).resolve(strict=True)
    target = _inside(root, output_dir)
    if not target.is_relative_to(root / "artifacts") or target.exists():
        raise ValueError("Choose a new cost output directory inside artifacts")
    spec_path = root / "configs/frozen-experiment.json"
    verified = verify_freeze(spec_path, root)  # Before audio decoding or model deserialization.
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    config = CostConfig.from_json(root / "configs/costs.json")
    spec = read("configs/frozen-experiment.json")
    manifest_name = "data/manifests/baseline-train-dev.csv"
    validate_manifest(root / manifest_name)
    with (root / manifest_name).open(newline="", encoding="utf-8-sig") as file:
        all_rows = [{key: row[key].strip() for key in REQUIRED_COLUMNS} for row in csv.DictReader(file)]
    if {r["split"] for r in all_rows} != {"train", "dev"}:
        raise ValueError("Cost benchmark requires the frozen train/dev manifest; evaluation is forbidden")
    rows = [r for r in all_rows if r["split"] == "dev"]
    if len(rows) != 700 or sum(r["label"] == "bonafide" for r in rows) != 100:
        raise ValueError("Require all 700 frozen development recordings")
    evidence_name = "artifacts/audits/day8/recordings.json"
    evidence_hash = _file_hash(root / evidence_name)
    if evidence_hash != read("docs/RECORDING_AUDIT_RESULTS.json")["local_artifact_sha256"]["recordings.json"]:
        raise ValueError("Recording audit evidence changed")
    evidence_rows = read(evidence_name)
    if [{k: r[k] for k in REQUIRED_COLUMNS} for r in evidence_rows] != all_rows:
        raise ValueError("Recording audit membership differs")
    evidence = {r["audio_path"]: r for r in evidence_rows}
    index = read("artifacts/features/baseline-index.json")
    indexed = {r["audio_path"]: r for r in index["records"]}
    old_scores_path = root / "artifacts/comparisons/day6-final/development_predictions.npz"
    if _file_hash(old_scores_path) != read("docs/METRIC_RESULTS.json")["source_development_scores_sha256"]:
        raise ValueError("Original development score artifact changed")
    with np.load(old_scores_path, allow_pickle=False) as saved:
        old_scores = {name: saved[name + "_scores"].copy() for name in MODELS}
        if not np.array_equal(saved["labels"], np.array([r["label"] == "spoof" for r in rows], dtype=np.int8)):
            raise ValueError("Original development label order differs")
    audio_config = AudioConfig.from_json(root / "configs/audio.json")
    feature_config = FeatureConfig.from_json(root / "configs/features.json")
    audio_root = (root / "data/asvspoof2019/LA").resolve(strict=True)
    waves, durations, details = [], [], []
    for row in rows:
        path = (audio_root / row["audio_path"]).resolve(strict=True)
        if not path.is_relative_to(audio_root):
            raise ValueError("Recording escapes the dataset root")
        wave, loaded = load_audio(path, audio_config)
        digest = waveform_digest(wave, audio_config.target_sample_rate, "<f4")
        if loaded["file_sha256"] != evidence[row["audio_path"]]["encoded_sha256"] or digest != evidence[row["audio_path"]]["processed_sha256"]:
            raise ValueError("Clean source differs from audited model input")
        wave.setflags(write=False)
        waves.append(wave); durations.append(len(wave) / audio_config.target_sample_rate)
        details.append({**row, "processed_frames": len(wave), "waveform_sha256": digest, "feature_sha256": indexed[row["audio_path"]]["vector_sha256"]})
    models = {name: joblib.load(_inside(root, spec["models"][name]["artifact"])) for name in MODELS}
    for model in models.values():
        parameters = model.get_params(deep=True)
        if any(value not in (None, 1) for key, value in parameters.items() if key == "n_jobs" or key.endswith("__n_jobs")):
            raise ValueError("Model requests parallel jobs beyond the single-thread procedure")
    records = []
    numerical_checks = {name: {"maximum_absolute_single_vs_batch_score_difference": 0.0,
                               "decision_difference_call_count": 0, "decision_difference_record_indices": set()} for name in MODELS}
    def log(stage, model, mode, repeat, position, count, seconds, elapsed):
        records.append({"stage": stage, "model": model, "mode": mode, "repeat": repeat,
                        "record_index": position, "rows": count, "audio_seconds": seconds, "elapsed_ns": elapsed})
    def notify(message):
        if progress is not None:
            progress(message)
    started_utc = datetime.now(timezone.utc).isoformat()
    hardware = hardware_identity()
    with threadpool_limits(limits=config.thread_limit):
        pools = [{k: entry[k] for k in ("user_api", "internal_api", "prefix", "version", "num_threads", "architecture", "threading_layer") if k in entry}
                 for entry in threadpool_info()]
        if any(entry["num_threads"] != 1 for entry in pools):
            raise ValueError("Native thread limit not honored")
        for wave in waves[:config.warmup_clips]:
            extract_features(wave, feature_config.sample_rate, feature_config)
        vectors = None
        for repeat in range(config.feature_repeats):
            current = []
            for position, wave in enumerate(waves):
                (vector, _), elapsed = measure_call(lambda: extract_features(wave, feature_config.sample_rate, feature_config))
                if _vector_hash(vector) != details[position]["feature_sha256"]:
                    raise ValueError("Timed feature differs from original model input")
                log("features", "shared", "one_clip", repeat, position, 1, durations[position], elapsed)
                current.append(vector)
            vectors = np.stack(current).astype(np.float64)
            notify(f"Feature pass {repeat+1}/{config.feature_repeats} verified all 700 vectors")
        # Materialize each row before timing; no feature/row construction is inside prediction calls.
        singles = [np.ascontiguousarray(vectors[i:i+1]) for i in range(len(rows))]
        def score(name, features):
            return model_scores(models[name], features, "margin" if name == "svm" else "probability")
        for name in MODELS:
            if not np.array_equal(score(name, vectors), old_scores[name]):
                raise ValueError("Warm batch scores differ from pre-freeze evidence")
            for i in range(config.warmup_clips):
                values = score(name, singles[i])
                if not np.allclose(values, old_scores[name][i:i+1], rtol=0, atol=1e-12):
                    raise ValueError("Warm single-clip score differs")
        for repeat in range(config.single_clip_repeats):
            order = MODELS[repeat % len(MODELS):] + MODELS[:repeat % len(MODELS)]
            for name in order:
                for i, features in enumerate(singles):
                    values, elapsed = measure_call(lambda: score(name, features))
                    if not np.allclose(values, old_scores[name][i:i+1], rtol=0, atol=1e-12):
                        raise ValueError("Timed single-clip score differs from original evidence")
                    numerical = numerical_checks[name]
                    numerical["maximum_absolute_single_vs_batch_score_difference"] = max(
                        numerical["maximum_absolute_single_vs_batch_score_difference"], float(np.max(np.abs(values - old_scores[name][i:i+1]))))
                    if not np.array_equal(threshold_predictions(values, spec["models"][name]["decision_rule"]),
                                          threshold_predictions(old_scores[name][i:i+1], spec["models"][name]["decision_rule"])):
                        numerical["decision_difference_call_count"] += 1
                        numerical["decision_difference_record_indices"].add(i)
                    log("inference", name, "one_clip", repeat, i, 1, durations[i], elapsed)
                notify(f"Single-clip inference {name}, pass {repeat+1}/{config.single_clip_repeats} verified")
        for repeat in range(config.batch_repeats):
            order = MODELS[repeat % len(MODELS):] + MODELS[:repeat % len(MODELS)]
            for name in order:
                values, elapsed = measure_call(lambda: score(name, vectors))
                if not np.array_equal(values, old_scores[name]):
                    raise ValueError("Timed batch scores differ from original evidence")
                log("inference", name, "batch_700", repeat, -1, len(rows), sum(durations), elapsed)
        notify("All seven full-batch inference calls per model verified")
    for numerical in numerical_checks.values():
        numerical["decision_difference_record_indices"] = sorted(numerical["decision_difference_record_indices"])
    verified = verify_freeze(spec_path, root)
    finished_utc = datetime.now(timezone.utc).isoformat()
    feature_records = [r for r in records if r["stage"] == "features"]
    summary = {"features": summarize_timings(feature_records), "models": {}}
    for name in MODELS:
        summary["models"][name] = {mode: summarize_timings([r for r in records if r["model"] == name and r["mode"] == mode]) for mode in ("one_clip", "batch_700")}
        summary["models"][name]["artifact_bytes"] = (root / spec["models"][name]["artifact"]).stat().st_size
    clock = time.get_clock_info("perf_counter")
    report = {"schema_version": 1, "scope": "warm-process component costs on 700 clean ASVspoof 2019 LA development clips; no evaluation audio",
              "started_utc": started_utc, "finished_utc": finished_utc, "config": asdict(config), "hardware": hardware,
              "native_threadpools": pools, "python": platform.python_version(), "audio_runtime": spec["audio_runtime"], "model_runtime": spec["model_runtime"],
              "timer": {"name": "time.perf_counter_ns", "implementation": clock.implementation, "resolution_seconds": clock.resolution, "monotonic": clock.monotonic,
                        "overhead_subtracted": False}, "gc_enabled": gc.isenabled(),
              "procedure": {"warmup": "first five manifest clips for features and each single model; one full-batch call per model",
                            "input_order": "original development manifest order in every pass; model order rotates by repeat",
                            "features": "complete frozen extract_features call on preloaded processed float32 waveforms; includes feature allocations",
                            "inference": "frozen model_scores including fitted pipeline and class/finite checks; contiguous float64 input already materialized",
                            "excluded": ["imports and warm-up", "audio acquisition/decoding/preprocessing/hashing", "codec/noise/bandwidth transformations", "cache lookup and model loading", "score comparison, threshold application and logging", "HTTP/UI, concurrency and queuing"],
                            "quantiles": "numpy linear on pooled measured calls; batch percentiles use only seven calls",
                            "timed_totals": "sum of measured call durations, not complete pass or job elapsed time",
                            "batch_amortization": "batch elapsed / 700 is throughput-derived cost, not one-clip latency"},
              "recordings": len(rows), "processed_audio_seconds": sum(durations), "processed_frame_range": [min(len(w) for w in waves), max(len(w) for w in waves)],
              "source_manifest_sha256": _file_hash(root / manifest_name), "source_recording_evidence_sha256": evidence_hash,
              "source_clean_scores_sha256": _file_hash(old_scores_path), "source_code_sha256": {"costs.py": _file_hash(Path(__file__))},
              "source_cost_config_sha256": _file_hash(root / "configs/costs.json"),
              "frozen_lock_sha256": verified["lock_sha256"], "frozen_file_bindings_verified": verified["file_bindings_verified"],
              "all_feature_vectors_match_original_evidence": True, "all_batch_scores_identical_to_original": True,
              "one_clip_score_absolute_tolerance": 1e-12, "one_clip_numerical_checks": numerical_checks,
              "one_clip_fixed_decisions_unchanged": all(v["decision_difference_call_count"] == 0 for v in numerical_checks.values()), "models_refitted": False, "thresholds_changed": False,
              "evaluation_audio_processed": False, "waveform_files_retained": False, "summary": summary}
    passes = pass_summaries(records)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tmp-costs-", dir=target.parent) as folder:
        staged = Path(folder)
        with (staged / "timings.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(records)
        with (staged / "passes.csv").open("w", encoding="utf-8", newline="") as file:
            fields = ["stage", "model", "mode", "repeat", "timed_calls", "processed_rows_with_repeats", "timed_total_ns", "amortized_ms_per_clip", "clips_per_timed_second", "processed_audio_seconds_with_repeats", "real_time_factor"]
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n"); writer.writeheader(); writer.writerows(passes)
        (staged / "recordings.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
        report["measured_calls"] = len(records); report["measured_passes"] = len(passes)
        report["local_artifact_sha256"] = {p.name: _file_hash(p) for p in staged.iterdir()}
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
        staged.rename(target)
    return report
