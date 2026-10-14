# Same-feature model comparison

Measured on 14 October 2026. This is development model selection on the ASVspoof 2019 LA subset, with known attacks A01-A06. No evaluation audio was acquired or processed.

## What stays comparable

Use the exact Day 5 manifest and feature index: 700 official training clips and 700 development clips, each containing 100 genuine clips and 100 clips from each of six spoof methods. Preserve official speaker-separated membership. All models see the same 40 MFCC summary features; speaker/attack IDs and filenames are not predictors. Input files, vector checksums, processing identity and membership are verified before fitting.

The majority and logistic baselines are refitted with their unchanged predefined settings. Their metrics and data hashes exactly match Day 5. Logistic is a fixed reference, while SVM and forest receive a small parameter search; this is not an equally extensive optimization of every algorithm.

## Search declared before scoring

[The configuration](../configs/comparison.json) declares ten candidates in this order:

- RBF SVM: C in [0.1, 1.0, 10.0], then gamma in [scale, 0.01]: six candidates. Float64 conversion and StandardScaler precede SVC. Every scaler and classifier fits training data only. SVC uses tolerance 0.001, at most 10,000 iterations, shrinking, and a 128 MiB kernel cache. Non-convergence aborts the run. Probability fitting stays disabled by the library default; use the decision margin directly.
- Random forest: maximum depth in [unlimited, 12], then minimum leaf size in [1, 3]: four candidates. Each uses 200 trees, Gini splits, sqrt feature sampling, bootstrap samples, seed 42 and one worker. Trees need no StandardScaler; the pipeline casts input to float64 and sklearn internally converts it to float32.

Both use balanced class weights derived from training counts. Logistic and forest emit spoof-class probabilities; SVM emits a signed decision margin, not a probability. Higher scores always mean spoof. Predefined decisions use score >= 0.5 for probability models and margin >= 0 for SVM; exact threshold ties count as spoof. Probabilities are not claimed to be calibrated. SVC's seed does not affect its deterministic fit when probability fitting is disabled; forest randomness uses seed 42.

Within each family select the highest development balanced accuracy; exact ties keep the first declared candidate. The overall development leader uses the same metric with logistic, SVM, forest as the tie order. There is no cross-validation or refit on training plus development. The runner rejects grids above 16 total candidates, forests above 200 trees, and unbounded SVM iteration settings. These limits keep this experiment manageable, not a promise about arbitrary dataset sizes.

## Actual results at untuned operating thresholds

Balanced accuracy averages genuine recall and spoof recall. Matrix counts below are TN/FP/FN/TP; spoof is positive.

| Model | Train balanced accuracy | Dev balanced accuracy | Dev raw accuracy | Dev false-positive rate | Dev spoof recall | Dev matrix |
| --- | --- | --- | --- | --- | --- | --- |
| Majority reference | 50.00% | 50.00% | 85.71% | 100.00% | 100.00% | 0/100/0/600 |
| Logistic regression | 87.50% | 78.92% | 83.86% | 28.00% | 85.83% | 72/28/85/515 |
| Selected RBF SVM | 87.75% | 84.25% | 80.14% | 10.00% | 78.50% | 90/10/129/471 |
| Selected random forest | 98.33% | 76.83% | 88.14% | 39.00% | 92.67% | 61/39/44/556 |

Selected SVM: C=0.1, gamma=scale. Selected forest: depth=12, minimum leaf=3. The SVM is the development leader on this declared metric. It falsely flags 10 genuine clips and misses 129 spoofs, versus logistic's 28 false positives and 85 missed spoofs. Its 5.33 percentage-point balanced-accuracy increase comes with lower spoof recall at these particular thresholds.

The forest's 88.14% raw accuracy looks strongest but hides 39 false positives among 100 genuine clips. Its 98.33% training balanced accuracy versus 76.83% development balanced accuracy also shows a substantial generalization gap. Different operating thresholds can change the tradeoffs; they have not been selected today.

## Complete candidate record

| Family | Zero-based index | Parameters | Dev balanced accuracy | Selected |
| --- | --- | --- | --- | --- |
| svm | 0 | C=0.1, gamma=scale | 84.25% | Yes |
| svm | 1 | C=0.1, gamma=0.01 | 83.92% |  |
| svm | 2 | C=1.0, gamma=scale | 82.25% |  |
| svm | 3 | C=1.0, gamma=0.01 | 83.00% |  |
| svm | 4 | C=10.0, gamma=scale | 72.42% |  |
| svm | 5 | C=10.0, gamma=0.01 | 81.67% |  |
| random_forest | 0 | max_depth=None, min_samples_leaf=1 | 71.42% |  |
| random_forest | 1 | max_depth=None, min_samples_leaf=3 | 76.17% |  |
| random_forest | 2 | max_depth=12, min_samples_leaf=1 | 75.67% |  |
| random_forest | 3 | max_depth=12, min_samples_leaf=3 | 76.83% | Yes |

[The machine-readable report](COMPARISON_RESULTS.json) contains every candidate's train/dev metrics, exact configuration, selection, decision rules, package versions, feature/data/source-code hashes and local artifact hashes. Fitted models and per-clip scores remain under ignored local artifacts. Load only your own trusted joblib files.

## Reproduce

Prepare the Day 5 audio/manifest/cache using [the baseline recipe](BASELINES.md), then run from the repository root:

```bash
python -m pip install -e ".[audio,models]"
python -m voicefake compare-models --index artifacts/features/baseline-index.json --manifest data/manifests/baseline-train-dev.csv --cache-dir artifacts/features/cache --output-dir artifacts/comparisons/day6-final --config configs/comparison.json
python -m unittest discover -s tests -v
```

Choose a fresh output directory for each run; existing runs are never overwritten. The local output contains four models, development scores in manifest order and report.json. The published report adds validation evidence after an independent repeated fit and reload check.

All 58 tests pass, including bounded search, exact selection ties, train-only SVM scaling, deterministic seeds, non-convergence rejection and model persistence. Repeating the real-data search reproduces every candidate metric and selected parameter. Selected development scores are exactly identical; predictions from all four restored models match within absolute tolerance 1e-12.

## Limits and next step

Development has now influenced parameter and model selection. These scores can be optimistic and cannot establish performance on unseen attack methods or current commercial generators. The 100 genuine development clips also give only one-percentage-point FPR resolution. No confidence interval, significance claim, low-FPR operating threshold, or final evaluation result is available yet. Decoded-audio duplicate auditing remains planned; current integrity checks cover metadata and identical encoded files.

The next milestone adds ROC-AUC, EER and a development-selected operating point. Model and threshold choices must be frozen before final evaluation.

## Implementation references

- [scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [scikit-learn RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn parameter-search guidance](https://scikit-learn.org/stable/modules/grid_search.html)
