# MFCC feature extraction and caching, version 1

The extractor converts processed audio into a fixed-length numeric vector. It does not fit a scaler, train a classifier, or score evaluation audio in this milestone. The recipe is stored in [configs/features.json](../configs/features.json) and uses the existing audio dependency group.

## Exact feature recipe

| Step | Setting |
| --- | --- |
| Input | Mono waveform at 16,000 Hz from the audio loader |
| Framing | 400 samples (25 ms), 160-sample hop (10 ms) |
| Boundaries | Complete frames starting at sample zero; no frame centering or signal padding; discard the final incomplete frame |
| Window | Periodic Hann: `0.5 - 0.5*cos(2*pi*n/400)` |
| Spectrum | 512-point real FFT; power `abs(FFT)**2 / 512` |
| Mel bands | 40 continuous-frequency triangles from 20 to 7,600 Hz, HTK spacing, peak scale 1, no area normalization |
| Log | Natural logarithm after flooring mel-band energy at `1e-12` |
| Cepstrum | Type-II orthonormal DCT, retain coefficients c0 through c19 |
| Summary | Mean of each coefficient, then population standard deviation (`ddof=0`) of each coefficient |
| Output | 40 float32 values: `mfcc_mean_00` through `mfcc_mean_19`, then `mfcc_std_00` through `mfcc_std_19` |

HTK spacing uses `mel(f) = 2595*log10(1 + f/700)`. Triangular weights are evaluated at the FFT-bin frequencies. The 400-sample frame is zero-extended for the 512-point FFT; this is spectral computation, not padding the recording to a longer duration. No pre-emphasis, delta coefficients, liftering, or per-utterance cepstral normalization is applied. The explicit log scale and filter normalization matter when reproducing the values; other library recipes can use different defaults.

The frame count is `1 + floor((samples - 400) / 160)`. The unused tail has fewer than 160 samples and is recorded. Silent frames remain finite because of the energy floor. Invalid or too-short waveforms fail explicitly. c0 retains energy-scale information; mean/std summaries lose the order of frames and cannot describe every temporal artifact in generated speech.

## Reproduce extraction

Install the audio dependencies and use a manifest containing recordings that are actually available locally. The day-3 local smoke manifest contains 14 official training/development clips.

```bash
python -m pip install -e ".[audio]"
```

Run this command as one line from the repository root:

```bash
python -m voicefake extract-features --manifest data/manifests/audio-smoke.csv --audio-root data/asvspoof2019/LA --cache-dir artifacts/features/cache --output artifacts/features/smoke-index.json --audio-config configs/audio.json --feature-config configs/features.json
```

The command writes local per-recording `features.npy` cache entries and a JSON index. The index preserves label, speaker, method, and partition metadata separately from the numeric vectors; those identifiers are not feature columns. Missing or invalid recordings fail the command instead of being silently removed. A partially completed run can leave valid reusable cache entries, but it does not write a completed index until every record succeeds. Existing indexes are not overwritten; use a new output name when repeating the command.

Evaluation input is rejected by default. The explicit `--allow-eval` option is reserved for processing the frozen final experiment; it is not used in today's check. No learned transform is fitted here. Future model pipelines must fit scaling or other learned preprocessing on training features only, and apply the fitted transform to development/evaluation features.

## Cache behavior

A cache key includes the encoded source-file SHA-256, semantic audio and feature configurations, extractor version, hashes of the loader/extractor/cache source code, and the relevant runtime/library versions. Filename, label, speaker ID, and partition do not enter the feature calculation or cache key. A changed input or processing recipe creates a different key.

Each entry contains a float32 vector plus metadata. Reads verify the identity, file checksum, vector checksum, shape, dtype, and finite values. Corrupt or incomplete entries fail explicitly; inspect and remove only the affected local cache entry before recomputing. NumPy loads use `allow_pickle=False`. A staged entry directory is renamed into place after both files are written, avoiding a normal interrupted write appearing as a complete cache entry.

Runtime identity is conservative: platform or code-byte changes can trigger a fresh entry even when the numerical output would be unchanged. Caches and indexes stay under ignored `artifacts/`. No cached features are uploaded to GitHub.

## Observed smoke check, 12 October 2026

All **42 tests pass**, including a separate slow DFT/cosine reference, an analytical amplitude-scaling check, silence flooring, summary dimensions, cache invalidation, corruption detection, and label separation.

The 14 official train/dev clips produced a **14 by 40 float32 matrix** with finite values. Their complete-frame counts ranged from **97 to 553**. The first pass had zero cache hits; the repeat had **14 hits**, with identical vector fingerprints. No evaluation audio was processed, no scaler was fitted, and no model was trained.

[FEATURE_SMOKE_AUDIT.json](FEATURE_SMOKE_AUDIT.json) records the configurations, source-code fingerprints, runtime, manifest provenance, matrix fingerprint, and actual cache observations. This is a pipeline check. Fourteen clips are insufficient for a useful detector benchmark; the training/development sample must be expanded before reporting model results.

Contains information from [ASVspoof 2019](https://doi.org/10.7488/ds/2555), made available under the [ODC Attribution License](https://opendatacommons.org/licenses/by/1.0/).

## References

- [SciPy DCT definitions and normalization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html)
- [Librosa mel-filter parameter conventions](https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html)
- [Librosa MFCC parameter conventions](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html)
