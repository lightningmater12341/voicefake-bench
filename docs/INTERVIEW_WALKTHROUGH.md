# Explaining VoiceFake Bench

Use this as a guide to the implemented work. Run the local development demo before presenting it, and distinguish what was measured from what remains unknown.

## A 45-second explanation

> I built a laptop-sized benchmark for detecting synthetic and voice-converted speech. I used ASVspoof 2019's speaker-separated partitions, reduced each recording to 40 MFCC statistics, and compared majority, logistic regression, SVM and random forest models. I selected the SVM and threshold on development data and froze the experiment before acquiring evaluation audio. On the clean held-out subset it detected 816 of 1,300 spoofs with one false alarm among 100 genuine clips. MP3 raised false alarms to 22, and two attack systems had only two detections each out of 100. The useful result is a reproducible account of where the detector fails.

## Walk through the experiment

1. **Question and data.** Ask whether a detector keeps working across different voices, generation methods and degraded recordings. Import official protocols and audit all 121,461 metadata rows. Keep official train/dev/eval assignments and speaker separation. SHA-256 ranking with seed 42 caps each split/label/attack stratum at 100: 700 train, 700 dev and 1,400 eval clips. This deliberately small subset differs from the full challenge and its natural class distribution.
2. **Representation.** Decode to mono 16 kHz with explicit duration rules. For each clip, compute 20 MFCC means and 20 population standard deviations. MFCCs summarize the spectral envelope; pooling keeps the model small but discards temporal structure. Fixed feature extraction fits no corpus statistics. The scaler inside the SVM/logistic pipeline fits training data only.
3. **Selection.** Keep logistic fixed and test six predefined SVM and four forest settings. Development balanced accuracy selects candidates and the primary SVM, with declared ties. The selected SVM uses RBF, C=0.1 and gamma=scale. These are development comparisons, not independent final performance estimates; search effort differs across model families.
4. **Decision threshold.** Using only 100 genuine development scores, allow at most one false alarm. Exclude the whole boundary tie group, save the next float above the first excluded score, and flag score >= threshold. The saved SVM margin threshold is 0.262431783372518. This empirical 1% target does not guarantee 1% on other recordings.
5. **Freeze and audit.** Seal models, configurations, thresholds, runtime and computational code before acquiring evaluation audio. Verify 36 bound files. Audit original membership and exact file/native/processed content across 2,800 selected recordings: no duplicate groups or excluded clips. Exact hashes do not rule out perceptual near-duplicates. Archive members passed CRC/length checks; the full archive was not downloaded or its full MD5 verified.
6. **Evaluation.** Run the frozen models once under clean, two noise levels, bandwidth reduction and MP3. Each condition starts separately from the same clean recording. Preserve scores/features and a completed journal; do not retune or replace the primary model. Bootstrap complete speaker blocks in 1,000 shared draws to estimate conditional uncertainty and paired changes.
7. **Costs and demo.** Separate warm feature extraction, one-clip inference and batch throughput. The local command uses exactly the frozen SVM recipe on development or personal audio and reports a margin and decision. It protects selected evaluation recordings from accidental reuse. It is an experimental demonstration, not a production authenticity service.

## The results to remember

These are the **primary SVM**, unchanged threshold, **100 genuine and 1,300 spoof** evaluation clips. Read full comparisons and uncertainty in [EVALUATION.md](EVALUATION.md).

| Condition | Genuine false alarms / 100 | Spoofs detected / 1,300 | Interpretation |
| --- | ---: | ---: | --- |
| Clean | 1 | 816 | 484 spoofs missed |
| Noise, 20 dB | 94 | 1,283 | Higher recall comes with severe false alarms |
| Noise, 10 dB | 100 | 1,300 | Every recording flagged |
| Bandwidth round trip, 8 kHz | 100 | 1,300 | Every recording flagged |
| MP3, 32 kbps | 22 | 951 | Threshold does not transfer reliably |

Clean ROC-AUC is 0.891500 and interpolated EER is 17.69%. The saved decision threshold is not the interpolated EER point. Bandwidth AUC is still 0.899254 while every genuine clip is falsely flagged: ordering scores well and making useful fixed-threshold decisions are different properties.

A17 and A18 each have 2/100 clean detections, A19 has 3/100, while A08/A09 each have 100/100. The evaluation includes eleven unknown systems and two known algorithm references: A16 shares the algorithm of training A04, and A19 that of A06. Attack IDs alone do not establish unseen algorithms.

Clean recall's 95% speaker-block interval is [58.43%, 66.83%]; genuine FPR's is [0.00%, 3.03%]. The MP3 FPR change is +21 percentage points, with paired interval [10.99, 31.31]. Those intervals keep the selected model/threshold fixed and do not describe all future generators or all dependencies within an attack method. Only 100 genuine clips inform these false-alarm rates.

## Show it in two minutes

Open the [README](../README.md), [frozen specification](../configs/frozen-experiment.json) and the evaluation table. Explain why the primary model and threshold were fixed first. Run the two **development** examples in [DEMO.md](DEMO.md); identify the score, threshold, hashes and audio-processing fields. Explain why a signed margin cannot be read as a confidence percentage. Return to the held-out report to show a failure as well as the aggregate result. The two demo examples do not estimate performance, and the command refuses selected evaluation content.

## Likely questions

**Why classical models?** They fit the laptop budget and make the representation, scaler, search and decision rule easy to inspect. They provide a reference for generalization experiments; this project does not claim state-of-the-art performance.

**Why not just accuracy?** Class counts differ, and a detector can get every spoof right by calling every recording spoof. Report false alarms, recall, confusion counts, ranking metrics and attack-specific failures together. Equal per-method caps also affect aggregate interpretation.

**How did you prevent leakage?** Preserve official speaker-separated partitions, fit on training only, select on development, check exact duplicates, then seal before evaluation. Save a single-run journal and verify independent metric calculations from its retained arrays. Reading saved scores to check arithmetic does not select a replacement model.

**Why choose SVM if forest later has a slightly higher evaluation AUC?** Development selected the primary model before evaluation. Switching now would use the reporting data for selection. Ranking and the chosen operating point also answer different questions.

**Why a speaker bootstrap?** Recordings from the same speaker are related. Resampling whole speakers retains that grouping and paired conditions share draws. It still cannot cover every dependence structure, attack-system uncertainty or selection uncertainty.

**Does noise prove MFCCs are the cause of failure?** No. The experiment identifies failures under declared transforms; it does not isolate a causal mechanism. Neither silence, amplitude nor pooling received a controlled causal ablation.

**Is it real time?** The measured warm SVM single-clip inference median is about 0.414 ms and MFCC extraction median about 1.784 ms on the documented laptop. Those are separate component timings and exclude decoding, model loading and application work. Batch amortized time is not request latency. The demo processes a complete recording; no streaming or end-to-end latency claim was tested. See [COSTS.md](COSTS.md).

**Will this catch today's cloned voices?** That was not tested. The corpus is from 2019, the experiment is a small declared subset, and digital perturbations do not reproduce a specific app/network. A model flag cannot prove a recording's origin.

## Find the implementation

| Concern | Source and evidence |
| --- | --- |
| Protocols and reproducible membership | `protocols.py`, [DATASET.md](DATASET.md) |
| Audio and MFCC recipe | `audio.py`, `features.py`, [FEATURES.md](FEATURES.md) |
| Bounded fitting and model choice | `baselines.py`, `comparison.py`, [MODEL_COMPARISON.md](MODEL_COMPARISON.md) |
| Thresholds and curve conventions | `metrics.py`, [METRICS.md](METRICS.md) |
| Content and frozen inputs | `audit.py`, `freeze.py`, [FROZEN_EXPERIMENT.md](FROZEN_EXPERIMENT.md) |
| Paired conditions and reporting | `perturbations.py`, `compression.py`, `robustness.py`, `evaluation.py` |
| Costs and local demo | `costs.py`, `demo.py`, [DEMO.md](DEMO.md) |

All fourteen planned milestones are complete. The repository includes aggregate results and reproducibility evidence; raw audio, cached features, model binaries and private build state stay outside Git.
