# Frozen experiment: voicefake-la-subset-v1

Sealed on **19 October 2026**, before evaluation audio acquisition, feature extraction or scoring. [The machine-readable lock](../configs/frozen-experiment.json) is the authoritative record. Its canonical content digest is:

```text
54ed0f229730513ea6aa6f5f405b358c2fae2e9e39428ad480a77d7d7a1ce346
```

The lock records choices; it contains no final evaluation results. Earlier result files retain their original pre-freeze status and dates. Future development robustness summaries may diagnose these fixed choices but must not retune the experiment.

## Fixed data and model choices

Preserve the original ASVspoof 2019 LA official partitions and speaker separation. Reproduce the Day 2 SHA-256 subset ranking with seed 42 and cap 100 per partition/label/attack group. The lock verifies the original source-key hashes and rederives the same selection.

| Partition | Genuine | Spoof | Purpose |
| --- | --- | --- | --- |
| Train | 100 | 600 | Fit existing models and scalers only |
| Development | 100 | 600 | Historical parameter and threshold selection |
| Evaluation | 100 | 1,300 | One later frozen reporting run, A07-A19 |

The ignored evaluation-only manifest is data/manifests/frozen-eval.csv. Only its metadata was created today, from the original selected records; it has 1,400 rows and 64 represented speaker IDs. Evaluation audio has not been downloaded or decoded. The small number of genuine examples limits false-positive-rate precision; this is a subset experiment, not a full challenge result or production guarantee.

The primary model is the **Day 6 selected RBF SVM**, with C=0.1, gamma=scale and training-fitted standardization. This follows the original development balanced-accuracy selection rule. The fixed comparison set also includes the majority reference, logistic regression and selected forest (200 trees, depth 12, minimum leaf size 3). Existing clean-trained model-file hashes are pinned; no refit on training plus development or noisy training augmentation is permitted.

## Unchanged clean-development operating points

Higher scores mean spoof; spoof=1 and bonafide=0. Apply score >= threshold unchanged to every condition and evaluation recording. Thresholds came from genuine clean development examples at a 1% empirical FPR target, with the documented whole-tie exclusion rule. They are not guaranteed to realize 1% FPR on evaluation or degraded data.

| Model | Score kind | Exact saved threshold |
| --- | --- | --- |
| majority | spoof_probability | 1.0000000000000002 |
| logistic | spoof_probability | 0.9898484358811777 |
| svm | svm_decision_margin | 0.262431783372518 |
| random_forest | spoof_probability | 0.7851178731435121 |

Report realized FPR, spoof recall and confusion counts, together with ROC-AUC and the existing tie-grouped linearly interpolated EER convention. EER is descriptive, not a replacement operating threshold. Report per-attack spoof recall with counts; no attack-specific threshold selection is allowed.

## Fixed conditions

Each starts independently from the unchanged clean mono/16 kHz/centre-crop recipe and the same 40 MFCC mean/std features:

- Clean: identity.
- Noise at 20 dB and 10 dB: content-seeded, paired realizations using the existing fixed recipe.
- Bandwidth: the existing 16 -> 8 -> 16 kHz filtered round trip with exact source length.
- MP3: 32 kbps mono, the pinned encoder/decoder, fixed gain 0.5, declared whole-frame zero extension, verified gapless decoding, removal only of that extension and inverse gain 2.0. No adaptive normalization or clipping.

Do not combine conditions or change their order, resample/crop policy, feature settings, model artifacts or thresholds after viewing evaluation. A codec headroom failure, metadata mismatch or other processing failure must be reported; it cannot silently exclude a recording or trigger a new gain/threshold.

## Analysis planned before evaluation

The next milestone implements **paired speaker-block bootstrap** uncertainty: 1,000 draws, seed 42, 95% percentile intervals with linear quantiles. Resample speaker blocks with replacement and keep all conditions for a source recording together. Do not refit models or reselect thresholds inside a draw. Exclude draws missing either class from interval calculation, report valid/invalid counts and do not replace them. These intervals quantify uncertainty across sampled speakers, not all possible future synthesis systems.

The evaluation content audit remains required before scoring. Keep one frozen reporting run after that audit, report actual clean/degraded results and failures, and do not tune based on them. Code for paired summaries, cost measurement, final scoring and the demo is still added in its later roadmap stage.

## Verify before later runs

```bash
python -m voicefake verify-freeze --root . --specification configs/frozen-experiment.json
```

The original lock verifies **36 file bindings**: critical preprocessing/model/metric/codec code, seven configurations, source audit evidence, local manifests and protocol keys, the feature index and four model artifacts. It also checks codec binary/build identity, audio runtime and model package versions. Model files are hashed without deserializing them. The CLI, README, learning notes and new reporting modules can evolve; frozen computational inputs cannot silently drift.

The lock's SHA-256 detects content edits, while its file/runtime bindings detect changes to experiment inputs. It is not a digital signature or a guarantee against somebody deliberately rewriting the lock. Git history and the published digest identify this original run.

A fresh clone intentionally lacks ignored audio, keys, caches and models. Reproduce the earlier recipes to obtain local artifacts; the original verifier will fail if exact bindings are absent or different. Another platform/build or regenerated model file may require a separately recorded freeze with its own content digest, not editing this published lock to force a pass.

To create a new snapshot after reproducing the declared setup, use fresh output and evaluation-manifest paths:

```bash
python -m voicefake freeze-experiment --root . --output configs/new-experiment.json --evaluation-manifest data/manifests/new-frozen-eval.csv
```

Creation refuses to overwrite an existing freeze or evaluation manifest. It verifies prior development and codec evidence, reconstructs metadata membership and seals choices without opening evaluation audio. The existing 105 tests cover content/threshold edits, file and runtime drift, repository path escapes and overwrite refusal; the actual original lock was also verified against every local binding.
