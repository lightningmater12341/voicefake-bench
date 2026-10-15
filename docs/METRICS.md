# Ranking metrics and a development operating point

Measured on 15 October 2026 from the saved Day 6 development scores. The four models, their settings, and their source scores are unchanged. No new model is selected or fitted in this milestone. All measurements below use 100 genuine and 600 spoof development clips, with known attacks A01-A06; evaluation audio remains untouched.

## Three different questions

ROC-AUC measures ranking across all thresholds. Here it is the probability that a randomly selected spoof receives a higher score than a genuine clip, counting an equal-score pair as half a win. Higher is better. It describes ranking on this sampled distribution; it does not measure calibration or guarantee a useful low-false-positive operating point.

Equal error rate (EER) describes where false-positive rate (FPR) and false-negative rate (FNR) meet. Lower is better. This implementation groups identical scores, includes the all-negative and all-positive endpoints, and linearly interpolates between adjacent ROC points at the FPR=FNR crossing. If the crossing is already an empirical point, no interpolation is needed. With ties or discrete samples, the interpolated rate may not be attainable by any single deterministic threshold. We report the bracket and interpolation weight, not an EER decision threshold. This is an explicit standalone convention, not a claim of matching official ASVspoof scoring code or t-DCF.

The operating point answers a narrower question: how many spoof clips are detected when the threshold allows at most 1% empirical FPR on genuine development clips? It is selected separately for each existing model, then saved for later unchanged application.

## Positive label and ties

Bonafide is 0 and spoof is 1. Higher scores always mean more evidence of spoofing. Logistic and forest produce spoof probabilities; SVM produces a decision margin. Their score magnitudes are not interchangeable, and probabilities have not been calibrated.

For ROC, lower the threshold through each distinct score, adding the whole equal-score block at once. Integrate TPR over FPR by trapezoids; this gives tied positive/negative pairs half credit. Reordering rows within a tied block cannot change the result. Scores that rank the classes backwards remain poor scores; the implementation never flips their orientation to improve AUC.

## Genuine-only threshold selection

[The configuration](../configs/metrics.json) fixes target_fpr=0.01 before measuring these results. The selector receives only genuine development scores:

1. Compute an integer false-positive budget as floor(decimal target_fpr * genuine_count). For 100 genuine clips, the budget is one; for 99 clips it would be zero. Decimal-to-rational arithmetic avoids an accidental downward rounding of an integer budget.
2. Sort genuine scores from highest to lowest. With budget b, use the score at zero-based index b as the exclusion boundary. For this run, that is the second-highest genuine score.
3. Save the smallest float64 value strictly above that boundary. Classify a future score as spoof when score >= the saved threshold. This excludes the whole boundary tie block rather than arbitrarily permitting some tied clips. Equal scores at the saved threshold count as spoof.

This is the most permissive representable float64 threshold satisfying the empirical budget. It does not inspect spoof scores or optimize their labels. Boundary ties may use less than the allowed budget. If no larger finite float64 exists, a serialized reject_all rule predicts bonafide for every finite score; JSON never contains Infinity or NaN. Applying a saved rule selects nothing and never recomputes a quantile from its new input.

## Actual development results

EER is interpolated; operating-point FPR and recall are actual integer-count measurements. Matrix order is TN/FP/FN/TP, with spoof positive.

| Model | ROC-AUC | EER | Selected-point FPR | Selected-point spoof recall | Selected-point matrix |
| --- | --- | --- | --- | --- | --- |
| Majority reference | 0.500000 | 50.00% | 0.00% (0/100) | 0.00% (0/600) | 100/0/600/0 |
| Logistic regression | 0.881267 | 21.00% | 1.00% (1/100) | 39.17% (235/600) | 99/1/365/235 |
| Selected RBF SVM | 0.902800 | 18.00% | 1.00% (1/100) | 66.83% (401/600) | 99/1/199/401 |
| Selected random forest | 0.918183 | 15.33% | 1.00% (1/100) | 64.17% (385/600) | 99/1/215/385 |

Random forest has the best overall AUC and lowest EER on this development sample, yet SVM detects more spoofs at the selected low-FPR operating point: 401 versus 385 of 600. Global ranking quality and performance in one small part of the ROC curve are different questions. The previously reported SVM leader by default-threshold balanced accuracy is retained as historical selection context; today's metric table does not silently replace the model-selection rule.

For SVM, moving from its default zero margin to this stricter threshold reduces false alarms from 10 to 1 among 100 genuine clips, while detections fall from 471 to 401 among 600 spoofs. That is 199 missed spoof clips at the selected operating point. A detector can look stronger on one measure while failing an important operating requirement.

The constant majority reference cannot separate the classes: AUC and EER are both 0.5. Its genuine scores are all 1, so the selected threshold lies just above 1 and flags nothing. Its zero false alarms come with zero spoof recall.

## Saved operating thresholds

These decimal representations round-trip exactly to float64. Use the stored decision rule, not a rounded display value.

| Model | Score kind | Saved threshold (score >= threshold means spoof) |
| --- | --- | --- |
| majority | spoof_probability | 1.0000000000000002 |
| logistic | spoof_probability | 0.9898484358811777 |
| svm | svm_decision_margin | 0.262431783372518 |
| random_forest | spoof_probability | 0.7851178731435121 |

[The machine-readable report](METRIC_RESULTS.json) contains EER brackets, thresholds, count-based metrics, source manifest/score/comparison hashes, referenced model hashes, settings, package versions and source-code hashes. The selected point is provisional development output; the full experiment freeze remains milestone 10.

## Reproduce and verify

First reproduce the [Day 6 comparison](MODEL_COMPARISON.md), then run from the repository root:

```bash
python -m voicefake score-development --run-dir artifacts/comparisons/day6-final --manifest data/manifests/baseline-train-dev.csv --output-dir artifacts/metrics/day7 --config configs/metrics.json
python -m unittest discover -s tests -v
```

Use a fresh output directory for each run. The metric command reads the source report, original train/dev manifest and checksummed development score archive. It verifies labels in manifest order, score orientation, finite numeric values and reproduction of the original default-threshold confusion metrics. It rejects evaluation membership. It does not deserialize models, load audio, refit anything or read feature values.

All 70 tests pass. Analytical cases cover perfect, reversed and constant rankings; a mixed tie block has EER 1/3; random tied curves match scikit-learn and independent pairwise AUC. Threshold tests cover decimal integer budgets, small samples, boundary ties, unchanged-rule application and JSON round trips. Artifact tests reject corruption, label misalignment and evaluation membership. On the actual saved scores, a second run produces an identical report, AUC/ROC match scikit-learn, and exhaustive empirical threshold checks confirm the selected recall is maximal within the one-false-positive budget. The published report adds that validation evidence to the runner's report.

## What this does and does not establish

These thresholds realize at most 1% FPR on the same genuine development sample used to select them. They do not establish a 1% population false-positive rate or promise it on unseen evaluation speakers. One error among 100 genuine clips is a coarse observation. The selected models were also chosen using development data; ranking and recall remain provisional, with no significance or uncertainty claim. Confidence intervals come in a later milestone.

Keep the stored thresholds unchanged when applying them to later data, and report the realized FPR and spoof recall there. Freeze model, preprocessing and threshold choices before final evaluation. Tomorrow's next milestone audits recording content and leakage; no final evaluation is run today.

## Metric API references

- [scikit-learn ROC convention and threshold orientation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [scikit-learn binary ROC-AUC score inputs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
