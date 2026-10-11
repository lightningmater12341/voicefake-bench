# Audio loading and processing, version 1

This milestone prepares waveforms. It does not extract ML features, train a detector, or score evaluation audio.

## Fixed recipe

The machine-readable recipe is [configs/audio.json](../configs/audio.json). The same loader and settings apply to both genuine and spoof recordings.

| Decision | Rule | Reason |
| --- | --- | --- |
| Decode | SoundFile/libsndfile to float32 samples | Read the corpus FLAC files without storing converted audio |
| Channels | Arithmetic channel mean to mono | One waveform shape for later feature extraction |
| Sample rate | 16,000 Hz | Consistent time/frequency scale and the corpus's native rate |
| Resampling | SciPy polyphase FIR, Kaiser beta 5, zero boundary extension | Low-pass filtering reduces aliasing when reducing sample rate |
| Minimum duration | Reject recordings shorter than 0.25 seconds | Avoid extremely short feature summaries; report failures rather than silently skip |
| Long recordings | Center-crop to at most 10 seconds after resampling | Bound later feature computation with a deterministic, label-independent rule |
| Short recordings | Keep actual duration; never pad to 10 seconds | Avoid introducing repeated samples or artificial silence |
| Silence | Retain original silent segments | Avoid an additional speech-detection algorithm and hidden removal rules |
| Volume | Preserve amplitude; no normalization or clipping | Keep preprocessing explicit and avoid a per-clip gain transform |
| Decode guard | Reject sources longer than 120 seconds before decoding | Bound ordinary source-duration handling |
| Invalid signals | Reject empty, non-finite, or zero-energy mono/output signals | Fail explicitly when a waveform cannot support meaningful processing |

A stereo signal whose channels cancel on averaging is rejected as zero-energy. Near-silent but nonzero signals are retained; this is not a voice-activity detector. Resampling can create peaks above the original range; the loader records the observed peak and does not clip it. Center cropping can discard useful speech or spoofing artifacts, so cropping is recorded for each clip. This recipe is a benchmark choice, not a claim of optimal preprocessing.

## Install and inspect

From the repository root:

```bash
python -m pip install -e ".[audio]"
python -m voicefake inspect-audio path/to/recording.flac --config configs/audio.json
python -m unittest discover -s tests -v
```

The audio dependency group adds NumPy, SciPy, and SoundFile. Metadata-only utilities still work with the base package. Running the full test suite requires the audio group. Install from a virtual environment; the exact versions used for the real smoke check are recorded in its audit.

`inspect-audio` prints JSON with source encoding, sample count/rate/channels, processing settings, resampling ratio, crop bounds, output duration, RMS and peak, source-file SHA-256, and processed float32 SHA-256. It produces no converted audio files. Internally, `load_audio` returns a one-dimensional waveform and this report; later models will use audio-derived features, not filenames or labels embedded in metadata.

Source and waveform hashes fingerprint this smoke check. Floating-point DSP results can vary across library versions and platforms; record those versions instead of assuming one processed hash is universal. Per-file hashing here is provenance, not the dataset-wide duplicate-content audit planned for a later milestone.

## Smoke-check scope

The local smoke input has one lexicographically first selected path for each training/development partition, label, and spoof-method stratum in the day-2 manifest: **14 official clips**. This includes two genuine clips and 12 spoof clips covering A01-A06 in each partition. No evaluation recordings are acquired or processed in this smoke check.

Official source ZIP members are acquired locally using bounded byte ranges and checked by the ZIP reader's CRC validation. Dataset terms and attribution are recorded in [DATASET.md](DATASET.md). Raw recordings and the smoke manifest remain in ignored `data/`; only a quality-control report is published. A 14-clip loader check does not establish corpus-wide validity or detector performance.

The synthetic unit fixtures exercise stereo averaging, native-waveform preservation, sample-rate conversion and alias suppression, crop positions, variable duration, invalid/zero signals, configuration validation, and FLAC decoding. Synthetic tones are test fixtures, not fake-speech benchmark data.

## Observed check, 11 October 2026

All 14 clips decoded successfully and produced identical waveforms and reports on repeat reads. They were mono, 16 kHz, FLAC PCM_16, with durations from **0.986 to 5.546 seconds**. No checked real clip required cropping or resampling; separate synthetic signal tests verify those code paths. The 14 locally retained files total 745,251 bytes. The full project test suite passes **30 tests**, and the dependency compatibility check passes.

[AUDIO_SMOKE_AUDIT.json](AUDIO_SMOKE_AUDIT.json) records the exact processing recipe, source and waveform fingerprints, acquisition scope, and runtime versions. These are loader observations, not detection results.

## References

- [SoundFile documentation](https://python-soundfile.readthedocs.io/en/latest/)
- [SciPy polyphase resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- [ASVspoof 2019 dataset](https://doi.org/10.7488/ds/2555)
