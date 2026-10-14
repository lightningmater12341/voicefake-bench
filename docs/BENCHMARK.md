# Experiment design, version 0.1

This design records the experiment boundaries. Measured development results are linked below; final evaluation remains prospective.

## Question and task

Classify genuine versus synthetic or voice-converted speech. Higher model scores mean more evidence of spoofing. The benchmark asks how classification quality changes for unfamiliar speakers, attack methods, and degraded audio.

Use ASVspoof 2019 LA as the starting corpus. It is an established research dataset, not a representative sample of every current voice generator, language, phone, or fraud scenario.

## Data boundaries

- Preserve official train/dev/eval membership and speaker separation.
- Store speaker ID and attack ID in manifests for auditing and breakdowns. Never use them or filenames as predictive features.
- If sampling, select deterministically within each original partition and label/attack stratum; record seed, selection rule, counts, and manifest hash.
- Run data-integrity checks before training. Reject shared recording IDs and speakers across partitions. Add content hashes during the audio-audit milestone.
- Fit scaling and any learned transformations on training data only.

A small subset is suitable for pipeline checks. It must not be described as a full ASVspoof result. Class balancing during sampling changes prevalence and must be reported.

## Models and selection

Start with a majority-class sanity baseline. Compare logistic regression, RBF SVM, and random forest on the same MFCC mean/std features. Specify sample rate, framing, feature count, silence treatment, and input-duration policy before running features. Log these choices in a configuration file.

The implemented [feature recipe](FEATURES.md) uses 20 MFCC means and 20 population standard deviations, with fixed 25 ms frames and 10 ms hops at 16 kHz. It fits no corpus statistics. Any later scaling must be fitted on training features only. The initial 14-clip check established the feature pipeline. The first [baseline experiment](BASELINES.md) now uses 700 training and 700 development clips from the original deterministic subset, with a training-fitted scaler and predefined logistic settings. Its results are development measurements; evaluation data stays unused until the later reporting milestone.

The implemented [model comparison](MODEL_COMPARISON.md) uses six predefined RBF SVM candidates and four random-forest candidates on the exact Day 5 data/features. Each fits training data only; development balanced accuracy selects settings, with first-declared-candidate ties. Logistic remains a fixed reference, so optimization effort differs. SVM decisions use signed margins at zero; probability-model decisions use 0.5, with threshold ties assigned to spoof. These operating points are untuned and development scores are provisional after selection.

Use a small predefined hyperparameter search and development data for model selection. Keep feature/model training separate from inference. Freeze configurations, model artifacts, and thresholds before evaluating the final partition.

## Metrics

Report ROC-AUC and equal error rate (EER), with explicit positive-label orientation and interpolation conventions. EER is a descriptive curve metric, not the deployed threshold.

Select a threshold targeting at most 1% false-positive rate on development genuine speech, using a documented tie policy. Report realized development FPR, fake recall, and the number of genuine clips. Carry that threshold unchanged to evaluation, reporting both evaluation FPR and fake recall. Limited sample sizes constrain the resolution of a 1% target.

Report per-attack fake recall with sample counts, plus runtime and hardware. A single aggregate metric can hide a detector that misses one generation method. Do not claim official challenge equivalence: t-DCF and the challenge's full operating assumptions are outside this first standalone-detector benchmark.

## Robustness conditions

Apply identical predetermined transformations to genuine and spoof clips. Keep every transformed version in its source partition. Planned conditions are additive seeded noise at 20 and 10 dB SNR, a documented low-bandwidth resampling round trip, and MP3 compression at a fixed bitrate. Digital perturbations are approximations; they are not measurements of any particular messaging app or telephone network.

Use clean-training models with unchanged thresholds initially. No tuning on evaluation or on its perturbed copies. Record codec versions and audio-transform settings. If a condition cannot be implemented reproducibly, label it unavailable rather than substituting an undocumented transformation.

## Uncertainty and reporting

Plan paired comparisons between clean and degraded scores for the same source clips. Resample by speaker for confidence intervals to respect within-speaker correlation; explain that speaker bootstrap intervals do not quantify uncertainty across all possible future generators.

Publish real experiment outputs with dataset provenance, exact commands, package versions, configuration hashes, model seeds, sample counts, and measured runtimes. Distinguish metadata fixtures, pipeline smoke checks, development results, and final evaluation results. Do not generate illustrative detection scores or backfill a results table.

## Sources

- [ASVspoof 2019 official dataset record](https://doi.org/10.7488/ds/2555)
- [ASVspoof 2019 challenge](https://www.asvspoof.org/index2019.html)
- [ASVspoof 2019 database paper](https://arxiv.org/abs/1911.01601)
