# Feature and inference costs

Measured on **21 October 2026**, using all 700 existing clean development clips, their frozen preprocessing and four saved models. The same subset contains 2399.224813 seconds of processed audio (about 39.99 minutes). This is a warm-process component benchmark, not a final evaluation or a service-latency test.

## Hardware and runtime

- CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz; 8 logical processors reported.
- System: Windows 11, build 10.0.26200, AMD64; Python 3.14.7.
- OS-reported usable physical memory: 15.75 GiB. This is hardware capacity, not measured peak process memory.
- AC power was connected when measurements started. CPU frequency, thermal state and Windows power plan were not pinned; other desktop activity was not isolated.
- Native OpenBLAS and OpenMP pools were verified at **one thread**. Model job settings were checked for incompatible parallelism. The [JSON evidence](COST_RESULTS.json) records package and native-library versions, including both loaded OpenBLAS builds.
- Models were already loaded; all source waveforms were decoded, processed, checked and held in RAM before timing. No evaluation audio was acquired or opened.

## Measured component costs

Each feature/single-clip distribution pools 2,100 measured calls: three passes over all 700 clips. Warm-up calls are excluded. Latency is wall time per complete call, in milliseconds; no timer-overhead subtraction is applied.

| Component, one clip per call | Median ms | p95 ms | Maximum ms | Mean ms per clip |
| --- | --- | --- | --- | --- |
| MFCC extraction, shared by every model | 1.7839 | 4.0814 | 7.1775 | 1.9980 |
| Majority reference prediction | 0.0125 | 0.0146 | 0.3401 | 0.0132 |
| Logistic regression prediction | 0.2807 | 0.3034 | 0.5850 | 0.2855 |
| Primary SVM prediction | 0.4144 | 0.5068 | 26.8512 | 0.4436 |
| Random forest prediction | 10.0606 | 10.5958 | 18.4718 | 10.1652 |

The forest costs much more per individual call than SVM or logistic here. Feature extraction also exceeds the primary SVM's median prediction cost. Timing prediction alone would omit a substantial part of that path. The 26.8512 ms SVM maximum remains in the evidence; no slow calls were discarded and this run does not establish their cause. Adding component medians or p95 values does not produce a measured end-to-end percentile.

For bulk prediction, each call processes the full 700-row float64 matrix. The table uses seven measured batch calls per model, after one warm-up batch. Batch cost per clip is total batch time divided by processed rows across those seven calls. It is a throughput measure, **not the response latency for a single clip**.

| Model | Median 700-clip call ms | p95 700-clip call ms | Mean batch cost ms/clip | Clips / timed second | Saved artifact KiB |
| --- | --- | --- | --- | --- | --- |
| Majority reference | 0.0493 | 0.0727 | 0.000070 | 14386377.0 | 0.57 |
| Logistic regression | 0.6874 | 0.7369 | 0.000950 | 1052744.7 | 2.78 |
| Primary SVM | 25.3412 | 26.4439 | 0.035477 | 28187.5 | 178.25 |
| Random forest | 17.5064 | 18.2135 | 0.024766 | 40377.7 | 1198.56 |

The forest's batch cost is lower than SVM's here, reversing their one-clip ordering. Call shape matters; this is not a universal ranking of algorithms. Seven batch calls give a rough comparison, not a reliable tail-latency service guarantee. Artifact size is serialized bytes on disk, not peak RAM or complete deployment size. The majority reference is a constant-score sanity check, not a useful detector at its selected threshold.

Feature extraction's measured real-time factor is **0.00058294**: sum of timed compute seconds divided by processed audio seconds, including repeats in both quantities. A value below one means this isolated computation consumed less time than the audio's duration on this run. This full-clip summary detector still needs the recording; the ratio does not establish streaming response time, throughput under load or an end-to-end real-time guarantee.

## Fixed measurement procedure

The [cost configuration](../configs/costs.json) was fixed before the measured run and does not alter the [sealed experiment](FROZEN_EXPERIMENT.md).

1. Verify all 36 frozen bindings and runtimes before decoding or deserializing models. Require the original train/dev manifest and select **all 700 development rows in original order**. Verify their encoded/processed hashes against the earlier recording audit.
2. Load clean processed float32 waveforms and the original fitted model artifacts. Hold inputs in memory; use no feature-cache hits for timed extraction. Materialize contiguous float64 model inputs before timing inference. The complete original model pipeline, including its training-fitted scaling where present, stays inside inference.
3. Warm feature extraction on the first five clips, then measure three complete feature passes. Verify every resulting vector hash against the original feature index outside its timer.
4. Warm each model on one full batch and the first five single clips. Measure three single-clip passes per model and seven full-batch calls per model. Rotate model order by repeat to avoid always timing one model last; clip order stays fixed. Cyclic rotation is not randomized scheduling or thermal control.
5. Time only the feature/scoring callable with time.perf_counter_ns. The scoring wrapper includes class-order and finite-output checks. Normal Python garbage collection remains enabled. Hashing, reference-score comparisons, threshold application, pass-loop bookkeeping, callbacks and log writes occur outside each timed call. Imports, acquisition, audio decoding/resampling/cropping, degradation transforms, cache lookup, model loading, network/UI, concurrency and queueing are excluded.
6. Record every positive integer duration, preserve all outliers, and use linear sample quantiles. Sum measured call durations for per-pass totals; these totals are not whole-pass or whole-job elapsed time. Check every batch score for exact agreement with the original run. Single-clip scores allow at most 1e-12 absolute drift, report observed differences and any decision changes, and never change the saved threshold. Reverify the freeze before writing outputs.

[Python's timer documentation](https://docs.python.org/3/library/time.html#time.perf_counter_ns) describes the high-resolution elapsed-time clock. This machine reports QueryPerformanceCounter with 100 ns resolution; integer nanosecond units do not imply 1 ns accuracy. [Scikit-learn's performance guide](https://scikit-learn.org/stable/computing/computational_performance.html) distinguishes prediction latency, bulk throughput and feature-extraction cost.

## Evidence and validation

All **126 tests pass**. The real run recorded **10,528 calls across 43 passes**: 2,100 feature calls, 8,400 single-clip prediction calls and 28 batch calls. An independent standard-library calculation reaggregated every raw duration, all pass totals, nine component groups, 18 median/p95 values, row-weighted throughput and audio-weighted real-time factors. The [43-row pass log](COST_RUNS.csv) is an exact copy of the local pass log; the [aggregate JSON](COST_RESULTS.json) pins its SHA-256 and the ignored raw-call log's hash.

All feature vectors and batch predictions match their original evidence exactly. Logistic's maximum single-versus-batch score difference was 4.440892098500626e-16; the other three models had zero difference. No fixed single-clip decision changed. The original 36 frozen bindings still verify; no model was refitted or threshold adjusted. These measurements add a cost axis; they do not repair the [development robustness failures](ROBUSTNESS.md) or select a new primary model.

## Reproduce

Prepare the earlier local data, model and audit artifacts; then run from the repository root:

```bash
python -m pip install -e ".[audio,models,compression]"
python -m voicefake verify-freeze
python -m voicefake benchmark-costs --root . --output-dir artifacts/costs/day12
python -m unittest discover -s tests -v
```

Use a new output directory under artifacts/ for another run. The runner refuses overwrite and writes timings.csv, passes.csv, recordings.json and report.json. Public evidence contains the aggregate report and pass timings only; raw per-call metadata, source identities, audio, feature arrays and model files stay local. Runtime measurements naturally vary between runs and machines; reproducibility here means a declared procedure, identifiable computational inputs, checked outputs and retained measurements, not identical clock values.
