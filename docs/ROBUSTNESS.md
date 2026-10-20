# Paired development robustness

This milestone applies the [frozen experiment](FROZEN_EXPERIMENT.md) to the same 700 development clips: 100 genuine, 600 spoof, known attacks A01-A06. These data already helped select models and thresholds. Results are development diagnostics conditional on those choices; held-out evaluation remains for the later reporting milestone.

## What stays paired and fixed

All four saved model files use their original clean-trained preprocessing and Day 7 clean-development decision rules. Every recording is processed independently as clean, noise at 20 dB, noise at 10 dB, 8 kHz bandwidth round trip, and 32 kbps MP3. Each degraded version starts from its clean waveform. There is no retraining, threshold adjustment or composition of degradations.

The runner verifies all 36 frozen bindings and runtime identities before decoding or model loading. It checks each source against the recording audit, every degraded waveform against the pre-freeze perturbation/codec fingerprints, and every regenerated clean feature and model score against its original evidence. Each model/condition prediction is computed twice and must match exactly. The freeze is verified again before writing outputs.

## Uncertainty procedure

Use the already frozen plan: 1,000 draws, PCG64 seed 42, 95% percentile intervals using NumPy linear quantiles. Sort represented speaker IDs, sample S speaker blocks uniformly with replacement S times per draw, and include every recording of each sampled speaker. A repeated speaker contributes its full block again; clips are not sampled independently. Metrics remain clip-weighted within each draw, so this is not a macro-average of speaker metrics.

Reuse the same draw for every condition and model. For each draw, calculate the metric change as degraded minus clean, then take the 2.5th and 97.5th percentiles of those paired changes. Subtracting endpoints from two separately computed intervals is not the procedure. Higher AUC/recall is better; lower EER/FPR is better. Changes are stored as fractions; tables may convert rates and changes to percent and percentage points.

Thresholds are applied unchanged inside every draw. Draws missing either class are excluded from intervals, counted and never replaced. If none are valid, intervals are null. Per-attack results contain actual spoof counts, detected counts and recall; per-attack intervals are not computed in this milestone.

[Scientific Python's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) describes shared-index pairing and percentile intervals. This implementation samples speaker blocks explicitly, rather than applying a clip-level bootstrap to correlated recordings.

## Measured development results — 20 October 2026

All 700 recordings completed all five conditions: 3,500 feature vectors and 20 model/condition prediction replays. Clean features and scores reproduced the original inputs and predictions exactly. Every degraded waveform matched its pre-freeze audit fingerprint. All 1,000 bootstrap draws across 20 represented speakers contained both classes; no draws were discarded. The [aggregate JSON](ROBUSTNESS_RESULTS.json) includes all model/condition intervals, paired changes, confusion matrices and per-attack counts.

The primary SVM remains the model selected by the original development balanced-accuracy rule. Its unchanged clean-development threshold is 0.262431783372518.

| SVM condition | ROC-AUC | EER % | Genuine false alarms / 100 | Spoofs detected / 600 | Spoof recall % |
| --- | --- | --- | --- | --- | --- |
| Clean | 0.902800 | 18.00 | 1 | 401 | 66.83 |
| Noise 20 dB | 0.587200 | 46.00 | 98 | 585 | 97.50 |
| Noise 10 dB | 0.479850 | 54.00 | 100 | 600 | 100.00 |
| Bandwidth 8 kHz | 0.862983 | 22.00 | 100 | 600 | 100.00 |
| MP3 32 kbps | 0.835867 | 22.67 | 11 | 439 | 73.17 |

Higher recall under degradation does not mean robustness improved. At 10 dB noise and under bandwidth reduction the SVM calls **every recording spoof**, including all 100 genuine clips. At 20 dB it makes 98 false alarms. Its clean threshold fails to transfer. Bandwidth AUC is still 0.862983 while its fixed operating point is unusable, showing why ranking quality and threshold behavior must both be reported.

MP3 produces 11 false alarms and detects 439 spoofs, versus one false alarm and 401 detections on clean audio. Pairing shows 10 new genuine false alarms, 61 newly detected spoofs and 23 lost spoof detections. That is a changed tradeoff, not a free detection gain. Attack-level counts expose another limitation: clean SVM detects only 6/100 A06 spoofs, falling to 2/100 after MP3, even while aggregate recall rises.

| SVM condition | FPR % [95% interval] | Spoof recall % [95% interval] |
| --- | --- | --- |
| Clean | 1.00 [0.00, 2.94] | 66.83 [58.82, 72.91] |
| Noise 20 dB | 98.00 [95.10, 100.00] | 97.50 [94.98, 99.80] |
| Noise 10 dB | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| Bandwidth 8 kHz | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| MP3 32 kbps | 11.00 [5.15, 17.55] | 73.17 [68.43, 77.12] |

| SVM degradation | Change in FPR, percentage points [paired 95% interval] | Change in recall, percentage points [paired 95% interval] |
| --- | --- | --- |
| Noise 20 dB | +97.00 [92.91, 100.00] | +30.67 [24.46, 38.97] |
| Noise 10 dB | +99.00 [97.06, 100.00] | +33.17 [27.09, 41.18] |
| Bandwidth 8 kHz | +99.00 [97.06, 100.00] | +33.17 [27.09, 41.18] |
| MP3 32 kbps | +10.00 [4.30, 16.81] | +6.33 [1.75, 11.28] |

The 100% rates with [100%, 100%] intervals mean every resampled block retains that observed all-spoof behavior. They do not prove a population rate of exactly 100% or guarantee perfect spoof detection on new data.

For all models, the entries below are **FPR % / spoof recall %**, using each model's own unchanged threshold. A low FPR accompanied by zero recall is also a failure.

| Condition | Majority reference | Logistic | Primary SVM | Forest |
| --- | --- | --- | --- | --- |
| Clean | 0.00 / 0.00 | 1.00 / 39.17 | 1.00 / 66.83 | 1.00 / 64.17 |
| Noise 20 dB | 0.00 / 0.00 | 0.00 / 0.00 | 98.00 / 97.50 | 53.00 / 66.67 |
| Noise 10 dB | 0.00 / 0.00 | 0.00 / 0.00 | 100.00 / 100.00 | 90.00 / 87.17 |
| Bandwidth 8 kHz | 0.00 / 0.00 | 12.00 / 63.67 | 100.00 / 100.00 | 23.00 / 82.33 |
| MP3 32 kbps | 0.00 / 0.00 | 0.00 / 14.67 | 11.00 / 73.17 | 7.00 / 65.00 |

Logistic calls every noisy clip genuine at its fixed threshold, missing all 600 spoofs. The majority reference also flags nothing under its genuine-only selected threshold. The forest's MP3 operating point has seven false alarms and 390 spoof detections; this comparison does not replace the frozen primary model or authorize selecting a new threshold.

## Validation

All **115 tests pass**. An independent check using scikit-learn reproduced ROC-AUC for all 20 model/condition combinations. A separate speaker-weight calculation, using block multiplicities rather than concatenated row indices, reproduced 80 rate and paired-change intervals. Every confusion matrix and per-attack count matched the saved scores. All 36 frozen bindings still verify. No evaluation audio was acquired or processed; no model was refitted or threshold changed.

## Limits on interpretation

The resampling assumes the sampled speakers approximate independent, exchangeable blocks. It does not account for all dependence shared across generation methods, corpus collection artifacts or future generators. No attack systems are resampled. Few speakers, only 100 genuine clips, a fixed small subset and percentile intervals limit inference, especially near rates of zero or one. A zero-width interval reflects the observed sample and fixed model; it cannot establish a zero population error rate.

These intervals are conditional on the selected model and thresholds. They do not repeat model/threshold selection or training, estimate uncertainty from the complete development process, or remove its selection bias. This is why the final frozen evaluation still matters. Inspecting these diagnostics must not trigger retuning the sealed experiment. Multiple reported metrics and conditions have no simultaneous-coverage correction; an interval excluding zero is not a blanket statistical significance claim.

## Reproduce

Prepare the earlier local recording/transform evidence, model files and freeze, then run from the repository root:

```bash
python -m pip install -e ".[audio,models,compression]"
python -m voicefake verify-freeze
python -m voicefake development-robustness --root . --output-dir artifacts/robustness/day11
python -m unittest discover -s tests -v
```

The runner refuses an existing output directory and requires its output under artifacts/. It writes report.json, scores.npz and recordings.json locally. Only an aggregate copy of the report is published; score arrays, per-record fingerprints, audio and model binaries stay ignored. Evaluation audio is never opened. A fresh clone must reproduce the earlier local artifacts before the original freeze can verify.
