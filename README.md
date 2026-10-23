# VoiceFake Bench

**Can a fake-voice detector survive unfamiliar voices, unseen generation methods, and degraded audio?**

A reproducible ML benchmark for distinguishing genuine speech from synthetic or voice-converted speech. The goal is to measure where detectors work and where they fail, rather than present a single impressive accuracy score.

## Completed project

All **14 milestones** are complete, with the final local demo and interview walkthrough added on **23 October 2026**. The [frozen evaluation](docs/EVALUATION.md) covers 1,400 held-out clips under five fixed conditions. Primary SVM detects **816/1,300 spoofs with 1/100 genuine false alarms on clean audio**; MP3 raises false alarms to **22/100**, while two hard attack systems have only **2/100 clean detections each**. The [local demo](docs/DEMO.md) reuses the frozen SVM and threshold. All 146 tests pass and the original [experiment freeze](docs/FROZEN_EXPERIMENT.md) still verifies.

Start with the [interview walkthrough](docs/INTERVIEW_WALKTHROUGH.md), [demo instructions](docs/DEMO.md), or the completed [roadmap](docs/ROADMAP.md).

## What was compared

- MFCC summary features with majority-class, logistic-regression, RBF SVM and random-forest models.
- Clean audio versus fixed noise, bandwidth and compression conditions, using the same recordings for paired comparisons.
- Overall results versus results for individual spoofing methods, with speaker-bootstrap uncertainty and measured component costs.

All models used the same data partitions and evaluation conditions. The SVM was selected on development data and stayed primary after evaluation.

## Dataset and evaluation

The starting dataset is **ASVspoof 2019 Logical Access (LA)**, which includes text-to-speech and voice-conversion attacks. Its official train, development, and evaluation partitions separate speakers, and evaluation includes previously unseen spoofing algorithms. See the [official dataset record](https://doi.org/10.7488/ds/2555).

We preserved those partitions, fitted preprocessing only on training data, selected models and decision thresholds on development data, then froze the experiment before scoring evaluation data. The laptop experiment uses 700 train, 700 development and 1,400 evaluation clips; these are subset results, not full challenge results.

Implemented metrics: ROC-AUC, linearly interpolated equal error rate, and fake-speech recall at a threshold selected from genuine development scores to target at most 1% empirical false-positive rate. The actual evaluation false-positive rate is reported separately. Accuracy alone can hide failures when class sizes differ.

The complete LA archive is about 7.12 GiB. [Dataset setup and terms](docs/DATASET.md) document storage, licensing, and a manageable subset strategy. Official metadata has been audited; audio is kept outside Git. The older dataset is a starting benchmark and does not establish performance on current commercial voice generators.

## Run the tools and local demo

Requires Python 3.10 or newer. Metadata tools use only the Python standard library. Audio loading and MFCC extraction use the optional audio group; the baseline runner also uses the models group.

From the repository root:

```bash
python -m pip install -e ".[audio,models,compression]"
python -m voicefake --help
python -m voicefake validate-manifest path/to/manifest.csv
python -m voicefake inspect-audio path/to/recording.flac --config configs/audio.json
python -m unittest discover -s tests -v
```

For the local inference demo on this completed experiment, follow [DEMO.md](docs/DEMO.md). It requires the original hash-verified model files and audit evidence, which are deliberately excluded from Git. A fresh clone plus dependency installation alone is insufficient.

Use `python -m voicefake import-protocols --help` for the importer and [the dataset recipe](docs/DATASET.md) for a complete command. It preserves official partitions, checks all source metadata before sampling, and writes a manifest plus a provenance sidecar. The default smoke subset contains 2,800 entries; it is not a model-performance result.

The [audio recipe](docs/AUDIO.md) defines mono conversion, 16 kHz resampling, silence/volume handling, and deterministic duration limits. The [real-clip audit](docs/AUDIO_SMOKE_AUDIT.json) records 14 successfully loaded official train/dev clips and repeatable decoding. The [feature recipe](docs/FEATURES.md) converts each clip into 40 MFCC summary values and caches them with provenance checks. The [feature smoke audit](docs/FEATURE_SMOKE_AUDIT.json) records a finite 14 × 40 matrix and 14 identical cache reuses. The [baseline recipe](docs/BASELINES.md) expands the real data to 1,400 clips and fits scaling on training features only. The [bounded model comparison](docs/MODEL_COMPARISON.md) reuses that exact sample and verifies every candidate and selected model. The [metric runner](docs/METRICS.md) audits those saved scores and records ranking metrics and development-selected thresholds. The [recording-content audit](docs/RECORDING_AUDIT.md) adds official-membership and exact-duplicate checks on all 1,400 local train/dev clips. The [perturbation property audit](docs/PERTURBATIONS.md) verifies fixed noise and bandwidth changes without modifying clean sources. The [MP3 audit](docs/COMPRESSION.md) adds a declared headroom and codec-boundary policy. The [freeze verifier](docs/FROZEN_EXPERIMENT.md) detects computational input or runtime drift. The [paired development report](docs/ROBUSTNESS.md) measures fixed-model behavior and speaker-bootstrap uncertainty across all five conditions. The [component cost runner](docs/COSTS.md) records warm feature and single/batch inference timings with verified outputs. The [evaluation audit and journaled runner](docs/EVALUATION.md) now report the frozen held-out results with per-attack failures and speaker-bootstrap uncertainty. The [local demo](docs/DEMO.md) verifies the frozen recipe before inference and rejects selected evaluation content; its development-only replay checks are recorded in [DEMO_VALIDATION.json](docs/DEMO_VALIDATION.json). All 146 tests pass; original model files, thresholds and earlier reports remain unchanged.

The manifest schema is:

```csv
audio_path,label,split,speaker_id,attack_id
train/flac/clip_001.flac,bonafide,train,speaker_001,-
dev/flac/clip_002.flac,spoof,dev,speaker_002,A01
eval/flac/clip_003.flac,spoof,eval,speaker_003,A07
```

These are illustrative filenames, not real dataset entries. The validator rejects missing fields, invalid labels, repeated paths, and speakers appearing in multiple partitions. The metadata validator does not read audio. The separate recording audit now checks exact content duplicates under different filenames or encodings and after preprocessing.

## Explain it in an interview

> I built a benchmark to test whether fake-voice detectors generalize across voices, generation methods and degraded audio. I froze a development-selected SVM and threshold before scoring evaluation. It detected about 63% of held-out clean spoofs at one false alarm among 100 genuine clips, but MP3 raised false alarms to 22%. The contribution is a reproducible evaluation protocol and an honest account of failure.

Use the [full interview walkthrough](docs/INTERVIEW_WALKTHROUGH.md) for a short pitch, demo steps and technical questions. The [benchmark design](docs/BENCHMARK.md) and [learning notes](docs/LEARNING_NOTES.md) explain the choices and tradeoffs.

## Attribution

ASVspoof 2019: Yamagishi et al., *The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database*, University of Edinburgh, 2019. Dataset DOI: [10.7488/ds/2555](https://doi.org/10.7488/ds/2555). The dataset has its own terms; no dataset audio is redistributed here.
