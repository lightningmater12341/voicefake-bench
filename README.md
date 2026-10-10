# VoiceFake Bench

**Can a fake-voice detector survive unfamiliar voices, unseen generation methods, and degraded audio?**

A reproducible ML benchmark for distinguishing genuine speech from synthetic or voice-converted speech. The goal is to measure where detectors work and where they fail, rather than present a single impressive accuracy score.

## Current stage

Day 2: official protocol import, provenance tracking, and reproducible subset manifests. No models have been trained and no detection results are available yet. This repository is being built in small, tested daily increments; the [roadmap](docs/ROADMAP.md) records what comes next.

## What will be compared

- MFCC summary features with logistic regression: an inexpensive, explainable baseline.
- The same features with an RBF support vector machine and a random forest.
- Clean audio versus fixed noise, bandwidth, and compression conditions, using the same recordings for paired comparisons.
- Overall results versus results for individual spoofing methods.

All models will use the same data partitions and evaluation conditions. A small neural model is a possible later extension, after the classical benchmark is complete.

## Dataset and evaluation

The starting dataset is **ASVspoof 2019 Logical Access (LA)**, which includes text-to-speech and voice-conversion attacks. Its official train, development, and evaluation partitions separate speakers, and evaluation includes previously unseen spoofing algorithms. See the [official dataset record](https://doi.org/10.7488/ds/2555).

We will preserve those partitions, fit preprocessing only on training data, select models and decision thresholds on development data, then freeze the experiment before scoring evaluation data. Subsets used for laptop experiments will be documented as subsets, not full challenge results.

Planned metrics: ROC-AUC, equal error rate, and fake-speech recall at a threshold selected to target a 1% false-positive rate on development data. The actual evaluation false-positive rate will also be reported. Accuracy alone can hide failures when class sizes differ.

The complete LA archive is about 7.12 GiB. [Dataset setup and terms](docs/DATASET.md) document storage, licensing, and a manageable subset strategy. Official metadata has been audited; audio is kept outside Git. The older dataset is a starting benchmark and does not establish performance on current commercial voice generators.

## Run today's component

Requires Python 3.10 or newer. The current metadata tools use only the Python standard library; audio/ML dependencies will be added with the components that need them.

From the repository root:

```bash
python -m pip install -e .
python -m voicefake --help
python -m voicefake validate-manifest path/to/manifest.csv
python -m unittest discover -s tests -v
```

Use `python -m voicefake import-protocols --help` for the new importer and [the dataset recipe](docs/DATASET.md) for a complete command. It preserves official partitions, checks all source metadata before sampling, and writes a manifest plus a provenance sidecar. The default smoke subset contains 2,800 entries; it is not a model-performance result.

The manifest schema is:

```csv
audio_path,label,split,speaker_id,attack_id
train/flac/clip_001.flac,bonafide,train,speaker_001,-
dev/flac/clip_002.flac,spoof,dev,speaker_002,A01
eval/flac/clip_003.flac,spoof,eval,speaker_003,A07
```

These are illustrative filenames, not real dataset entries. The validator rejects missing fields, invalid labels, repeated paths, and speakers appearing in multiple partitions. It does not yet read audio or identify duplicate recordings stored under different filenames.

## Explain it in an interview

> I'm building a benchmark that asks whether fake-voice detectors generalize to unfamiliar voices and spoofing methods. I start with understandable audio features and classical models, use separate data for training and model selection, and then test fixed models on clean and degraded audio. The contribution is the evaluation protocol and failure analysis; results are added only after the experiments run.

See [the benchmark design](docs/BENCHMARK.md) and [learning notes](docs/LEARNING_NOTES.md) for the reasoning behind each choice.

## Attribution

ASVspoof 2019: Yamagishi et al., *The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database*, University of Edinburgh, 2019. Dataset DOI: [10.7488/ds/2555](https://doi.org/10.7488/ds/2555). The dataset has its own terms; no dataset audio is redistributed here.
