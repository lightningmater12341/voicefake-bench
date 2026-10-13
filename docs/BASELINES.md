# First train/development baselines

This milestone compares a majority-class reference with a simple learned detector. Results below are actual measurements on a documented **ASVspoof 2019 LA subset**, using the official train/dev partition membership. These development results do not measure unseen-generation methods, degraded audio, or the final evaluation partition.

## Data and model recipe

The subset contains **700 training and 700 development clips**: 100 genuine recordings and 100 recordings from each of A01-A06 in each partition. It retains every train/dev entry from the earlier seed-42, cap-100 manifest, with no additional sampling. All 1,400 clips passed archive-member CRC checks. The locally retained audio totals 86,733,549 bytes; bounded archive requests transferred 100,387,861 bytes. No evaluation audio was acquired or processed.

The source manifest reproduces byte-for-byte from the published selection recipe. [BASELINE_RESULTS.json](BASELINE_RESULTS.json) records its fingerprint, official key fingerprints, acquisition scope, configurations, feature provenance, runtime versions, and measured metrics. Source audio, manifests, cached features, model binaries, and per-recording scores remain local.

| Component | Fixed setting |
| --- | --- |
| Numeric inputs | The existing 40 MFCC mean/std summary values |
| Majority reference | Predict the most frequent training class for every clip |
| Learned detector | Float64 conversion, StandardScaler, then L2 logistic regression |
| Scaling | Fit mean/std on the 700 training rows only; use the fitted transform on development |
| Logistic settings | C=1.0, lbfgs, max_iter=2,000, tolerance=0.0001, class_weight=balanced, seed=42 |
| Class weighting | Inverse training-class frequency; development labels do not affect weights |
| Decision rule | Classify as spoof when its model score is at least 0.5; ties count as spoof |
| Model selection | No search or development-based parameter changes in this milestone |

The model settings are in [configs/baseline.json](../configs/baseline.json). Logistic regression converged in **26 iterations**. The saved pipeline includes input-precision conversion so later inference uses the same calculation as training. Its scores are model outputs; class weighting and this experiment do not establish calibration as fraud probabilities.

## Development results, 13 October 2026

Each model is scored on the same 700 development clips. Genuine speech is label 0 and spoofed speech is label 1. Balanced accuracy is the mean of genuine recall and spoof recall.

| Model | Accuracy | Balanced accuracy | Genuine recall | Spoof recall | Genuine false-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority class | 85.71% | 50.00% | 0.00% | 100.00% | 100.00% |
| Logistic regression | 83.86% | 78.92% | 72.00% | 85.83% | 28.00% |

Logistic development confusion matrix, with true labels as rows and predicted labels as columns:

| True class | Predicted genuine | Predicted spoof |
| --- | ---: | ---: |
| Genuine | 72 | 28 |
| Spoof | 85 | 515 |

The majority reference calls all 700 clips spoof: it correctly counts the 600 spoof examples and falsely flags all 100 genuine examples. Its 85.71% accuracy comes from class proportions. Logistic regression has lower raw accuracy on this subset but distinguishes both classes, reaching 78.92% balanced accuracy. At the current untuned threshold, **28 of 100 genuine development clips are falsely flagged**. Improving and understanding that failure is part of the benchmark.

Logistic training balanced accuracy is 87.50%; development balanced accuracy is 78.92%. These are descriptive values for this sample and recipe, not a causal explanation for the gap. Complete train/dev metrics, including spoof precision/F1 and confusion matrices, are in the JSON report. Undefined precision when no clips are predicted positive is recorded as zero by convention.

## Reproduce

Start with the original local seed-42, cap-100 source manifest and the official audio files referenced by it. For a standard download, extract the LA archive as described in [DATASET.md](DATASET.md). Today's bounded acquisition was a local preparation step; the filtering script and model commands do not download speech.

```bash
python -m pip install -e ".[audio,models]"
python scripts/prepare_baseline_manifest.py --source data/manifests/smoke.csv --output data/manifests/baseline-train-dev.csv
```

Run each following command as one line from the repository root:

```bash
python -m voicefake extract-features --manifest data/manifests/baseline-train-dev.csv --audio-root data/asvspoof2019/LA --cache-dir artifacts/features/cache --output artifacts/features/baseline-index.json --audio-config configs/audio.json --feature-config configs/features.json
```

```bash
python -m voicefake train-baselines --index artifacts/features/baseline-index.json --manifest data/manifests/baseline-train-dev.csv --cache-dir artifacts/features/cache --output-dir artifacts/baselines/day5 --config configs/baseline.json
```

Use a fresh filename or run directory when an output already exists. The runner checks source-manifest membership, feature/index hashes, common processing identities, speaker separation, and encoded-file overlap across train/dev before fitting. It requires both classes in both partitions, with a default minimum of 20 examples per class. This minimum is a pipeline guard, not a guarantee of statistical adequacy.

The local run directory contains `report.json`, saved majority/logistic models, and development labels/scores. Both models use the same data. A repeated fit produced identical metrics and development scores; a reloaded pipeline reproduced the stored scores. The full test suite passes **51 tests**, and the dependency check passes. The tests include training-only scaling under shifted development inputs, label-membership tampering, evaluation-input rejection, corrupted feature files, and model persistence.

## Boundaries and next work

This is a small development experiment with known A01-A06 attack methods and a sampled class distribution. With only 100 genuine development clips, a single false positive changes the measured rate by one percentage point. A low-FPR threshold has not yet been selected. ROC-AUC/EER, confidence intervals, degraded-audio experiments, and final frozen evaluation remain later milestones.

Encoded-file overlap is checked here; a broader decoded-content duplicate audit is still planned. These results establish a measured starting point for the next same-data model comparison. They do not establish performance on current voice generators or real fraud calls.

Contains information from [ASVspoof 2019](https://doi.org/10.7488/ds/2555), made available under the [ODC Attribution License](https://opendatacommons.org/licenses/by/1.0/).

## References

- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- [scikit-learn preprocessing and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
