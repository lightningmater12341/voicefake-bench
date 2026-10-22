# Frozen evaluation and failure analysis

This is the one frozen report for voicefake-la-subset-v1, whose choices were sealed on 19 October 2026. The held-out subset contains 100 genuine clips and 100 spoof clips for each attack ID A07-A19: **1,400 recordings across 64 represented speaker IDs**. Official train/dev/eval membership and full-key speaker separation are preserved. These are subset measurements, not full-challenge results, t-DCF scores or evidence about current commercial voice generators.

The primary model remains the original selected SVM (C=0.1, gamma=scale), with clean-development margin threshold 0.262431783372518. The saved majority, logistic and forest comparisons retain their own original thresholds. No parameters, scalers, model files or thresholds are fitted/selected using evaluation. All conditions start separately from the fixed clean processed waveform.

## Provenance and pre-score content checks

The original cap-100/seed-42 metadata selection was reconstructed and frozen before evaluation audio acquisition. The exact evaluation-manifest SHA-256 is eb8100d17dff341e4ea81f8890a51613eb40848089675218716d5e28dceb6543. [Acquisition evidence](EVALUATION_ACQUISITION.json) records 1,400 selected clips, 85,358,353 extracted audio bytes and 99,635,760 HTTP-range bytes, including archive directory/member overhead. All selected members passed ZIP CRC/length checks. The complete 7.12 GiB archive was not downloaded and its full MD5 was not verified. Attribution and license are retained under the [dataset recipe](DATASET.md); raw audio stays ignored.

Before any detector scoring, all 1,400 evaluation recordings were checked against their official keys and acquisition hashes. The earlier 1,400 train/dev source files were rehashed against their original evidence. The combined selected set of 2,800 recordings had **zero exact duplicate groups** at encoded-file, native-decoded and processed-waveform levels. There were no audit failures or removals. One evaluation clip used the already frozen centre-crop limit; none needed resampling. [Content-audit evidence](EVALUATION_CONTENT_AUDIT.json) pins the original and new fingerprints. Exact hashing does not detect shifted, gain-changed or perceptually similar near-duplicates.

Evaluation contains eleven unknown attack systems and two known algorithm references: **A16 uses the A04 algorithm, and A19 uses the A06 algorithm**. Distinct IDs do not make all thirteen algorithms unseen. This distinction follows the [dataset authors' description, section 3.1](https://arxiv.org/pdf/1911.01601). Per-attack counts remain explicit rather than hiding these systems in an aggregate.

## Measured results — 21 October 2026

The frozen run completed all **1,400 recordings × five conditions = 7,000 feature vectors** with no processing failures or removals. All four saved models were scored on the same paired inputs. Every one of the 1,000 speaker-bootstrap draws over 64 represented speaker IDs contained both classes. The [aggregate result JSON](EVALUATION_RESULTS.json) preserves all model/condition intervals, paired changes, confusion counts and per-attack counts.

| Primary SVM condition | ROC-AUC | EER % | Genuine false alarms / 100 | Spoofs detected / 1,300 | Spoof recall % |
| --- | --- | --- | --- | --- | --- |
| Clean | 0.891500 | 17.69 | 1 | 816 | 62.77 |
| Noise 20 dB | 0.695062 | 34.08 | 94 | 1283 | 98.69 |
| Noise 10 dB | 0.577454 | 41.00 | 100 | 1300 | 100.00 |
| Bandwidth 8 kHz | 0.899254 | 18.00 | 100 | 1300 | 100.00 |
| MP3 32 kbps | 0.803138 | 25.77 | 22 | 951 | 73.15 |

Clean SVM detects 816 spoofs and misses **484**, while falsely flagging one genuine clip. The observed 1% FPR happens to match its development rate on this small held-out genuine sample; it does not establish a production guarantee. The clean recall interval is [58.43%, 66.83%], and the FPR interval is [0.00%, 3.03%], conditional on the declared speaker resampling.

MP3 increases recall from 62.77% to 73.15% but increases genuine false alarms from one to **22**. Pairing shows 144 newly detected spoofs, nine lost detections and 21 new genuine false alarms. Its FPR change is +21 percentage points, paired 95% interval [10.99, 31.31]; its recall change is +10.38 points, interval [8.10, 12.54]. This is a worse false-alarm tradeoff, not free robustness.

The SVM calls **all 1,400 clips spoof** at 10 dB and under bandwidth reduction. The latter still has AUC 0.899254, slightly above clean AUC, while its unchanged operating threshold fails completely on genuine speech. Relative ranking and a usable decision point are different properties. The 100% recall and [100%, 100%] resampling intervals in those conditions reflect observed all-spoof decisions; they do not demonstrate a useful detector or exact population certainty.

| SVM condition | FPR % [95% interval] | Spoof recall % [95% interval] |
| --- | --- | --- |
| Clean | 1.00 [0.00, 3.03] | 62.77 [58.43, 66.83] |
| Noise 20 dB | 94.00 [89.52, 98.02] | 98.69 [97.99, 99.36] |
| Noise 10 dB | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| Bandwidth 8 kHz | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| MP3 32 kbps | 22.00 [12.15, 31.82] | 73.15 [69.03, 76.73] |

## Attack-level failure analysis

Each entry below is the number of SVM spoof detections **out of exactly 100 clips for that attack**. A16 and A19 are the known algorithm references described above; the other eleven are unknown systems relative to training/development. All attacks reach 100/100 detections under 10 dB and bandwidth changes, accompanied by 100/100 genuine false alarms, so those two columns would not demonstrate attack robustness.

| Attack | Clean detected / 100 | MP3 detected / 100 | Noise 20 dB detected / 100 |
| --- | --- | --- | --- |
| A07 | 88 | 93 | 100 |
| A08 | 100 | 100 | 100 |
| A09 | 100 | 100 | 100 |
| A10 | 72 | 87 | 100 |
| A11 | 91 | 95 | 100 |
| A12 | 50 | 81 | 100 |
| A13 | 89 | 99 | 98 |
| A14 | 95 | 100 | 97 |
| A15 | 91 | 95 | 99 |
| A16 (known reference) | 33 | 76 | 98 |
| A17 | 2 | 17 | 96 |
| A18 | 2 | 2 | 100 |
| A19 (known reference) | 3 | 6 | 95 |

The aggregate clean recall hides **2/100 detections for A17, 2/100 for A18 and 3/100 for A19**. These three voice-conversion systems are difficult for this fixed model in this subset. A19 is a known algorithm reference, so unfamiliar algorithm identity alone cannot explain every miss. Conversely, the model detects all 100 clean A08 and A09 spoofs, which are unknown systems. “Unknown” is not a universal difficulty ranking.

These results describe failure patterns. They do not isolate a causal mechanism in MFCC pooling, silence, amplitude, particular speakers or a generator component. No post-evaluation ablation, threshold adjustment or alternative model search was performed. Development/evaluation also differ in voices and attack composition, so their aggregate gap cannot be attributed only to unknown generators.

## Fixed comparison models

Entries are **realized FPR % / spoof recall %**, applying each saved clean-development threshold unchanged.

| Condition | Majority reference | Logistic | Primary SVM | Forest |
| --- | --- | --- | --- | --- |
| Clean | 0.00 / 0.00 | 1.00 / 35.00 | 1.00 / 62.77 | 5.00 / 57.00 |
| Noise 20 dB | 0.00 / 0.00 | 0.00 / 0.69 | 94.00 / 98.69 | 40.00 / 64.08 |
| Noise 10 dB | 0.00 / 0.00 | 0.00 / 0.00 | 100.00 / 100.00 | 81.00 / 88.54 |
| Bandwidth 8 kHz | 0.00 / 0.00 | 3.00 / 53.85 | 100.00 / 100.00 | 21.00 / 68.08 |
| MP3 32 kbps | 0.00 / 0.00 | 0.00 / 12.23 | 22.00 / 73.15 | 9.00 / 64.69 |

Forest's clean AUC is 0.893746, marginally above SVM's, but its saved operating point flags five genuine clips and detects 741 spoofs. SVM flags one and detects 816. Logistic misses all 1,300 spoofs at 10 dB; the majority reference flags nothing under its genuine-only selected threshold. Comparisons remain visible without replacing the original primary model based on this held-out report.

## Validation and interpretation limits

All **136 tests passed before scoring**. The separate saved-score validation reproduced 20 AUCs using scikit-learn, 20 interpolated EERs using an independently generated ROC, all confusion counts and 260 per-attack count entries. A separate speaker-weight calculation reproduced 80 rate/paired-change intervals. The completed claim and all score/feature artifact hashes verify, as do the original 36 frozen bindings. No raw-audio rescoring, model refit or threshold reselection was needed for these checks.

Only 100 genuine clips constrain FPR precision. The cap-100 attack sampling changes original corpus prevalence; aggregate recall weights the thirteen methods equally here. Speaker-block intervals do not resample generator algorithms or fully describe their dependence. Exact duplicate checks do not rule out near-duplicates. These 2019 systems and controlled digital transformations do not establish performance on today's generators or any particular messaging app. No official full-dataset or production claim follows from this small experiment.

## Run and uncertainty procedure

Verify all 36 original frozen file bindings and runtime identities. Require a passing matching content audit before feature processing or model loading. Check each evaluation source again against its audited file and processed-waveform hashes. Extract the same 40 MFCC summary features for clean, 20/10 dB noise, bandwidth reduction and 32 kbps MP3, without training augmentation, clipping or adaptive gain. A processing failure is logged and blocks scoring; it cannot silently drop a recording or change the declared codec recipe.

A claim journal permits only one logical run for this frozen specification, even if a different output directory is requested. Features, score arrays and their hashes are retained locally as checkpoints. An interrupted run may resume only with the same frozen digest, audit, manifest, output path and evaluation/summary code. A completed claim refuses another run; metric validation uses saved score arrays, without selecting anything or producing another fitted experiment. This is an operational guard, not a security boundary against deliberate local-file edits.

Apply the saved score >= threshold rules unchanged. Report ROC-AUC, the existing tie-grouped interpolated EER, realized FPR, spoof recall, confusion counts and per-attack recall/counts. Thresholds are never recomputed from evaluation genuine examples. A descriptive EER is not an operating threshold.

Use the previously frozen uncertainty plan: 1,000 shared PCG64 seed-42 speaker-block draws, 95% percentile intervals with linear quantiles, no threshold reselection and explicit invalid-draw counts. Every sampled speaker contributes its whole block, including repeated-speaker multiplicity. Metrics remain clip-weighted. Every model/condition shares draws, and degraded-minus-clean changes are computed within a draw. These intervals condition on existing model files and thresholds, assume exchangeable speaker blocks, and do not quantify arbitrary future systems or all shared generator/corpus dependence. Multiple reported intervals have no simultaneous-coverage correction.

## Reproduce or inspect the saved run

Preserve the original local data, keys, model artifacts and freeze. After obtaining precisely the frozen evaluation membership and recording its archive provenance:

```bash
python -m voicefake verify-freeze
python -m voicefake audit-evaluation --root . --output-dir artifacts/evaluation-audit/day13
python -m voicefake evaluate-frozen --root . --audit-dir artifacts/evaluation-audit/day13 --output-dir artifacts/evaluation/frozen-v1
```

If execution was interrupted, use evaluate-frozen with --resume and the same paths/code; a completed run is read from its existing report and scores instead. The content audit refuses overwrite as well. Another reproduction must use its own checkout/local artifacts and report its own identity; the published original is never rewritten to make a changed environment pass.

Source audio, per-record fingerprints, cached features, individual score arrays and the claim journal stay local. Public files contain aggregate provenance, content checks, metrics and failure analysis only. Evaluation results describe what this frozen baseline does; later demo work must keep these choices and limitations visible.
