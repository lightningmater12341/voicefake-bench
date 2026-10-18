# Fixed noise and bandwidth recipes

Measured on 18 October 2026. This milestone implements reproducible waveform changes and checks their properties on all 1,400 existing train/dev clips. It does not compute detector scores, refit models, change thresholds or process evaluation audio. Compression and the experiment freeze remain milestone 10.

## Shared rules

Start with the existing clean recipe: mono float32 audio at 16 kHz, already resampled/cropped according to [the audio configuration](../configs/audio.json). Apply each condition separately to that clean waveform; noise and bandwidth conditions are not stacked. Preserve the clean source, sample count and source partition. The same functions apply to genuine and spoof recordings without inspecting labels or speaker/attack IDs.

[The perturbation configuration](../configs/perturbations.json) declares seed 42, noise levels 20 and 10 dB, and the 8 kHz intermediate rate. No settings were selected using detector scores. The clean extractor, caches and model pipelines are unchanged. No transformed waveform is written to disk in today's property audit; per-record measurements and fingerprints stay in ignored artifacts.

## Noise at a measured SNR

Generate a float64 standard-normal realization with an explicit NumPy Generator(PCG64). The seed is the integer value of SHA-256 over compact JSON containing the domain voicefake-noise-v1, base seed and clean waveform fingerprint. That fingerprint includes sample rate, shape and canonical samples. Filenames, labels, directory traversal order and global random state cannot influence the draw. Re-encoding identical clean audio does not change its noise identity.

Both SNR conditions use the same noise direction; their amplitudes differ. For each clip, scale the realized noise using:

```text
signal_power = mean(clean_float64 ** 2)
noise_scale = sqrt(signal_power / (10 ** (target_snr_db / 10) * mean(raw_noise ** 2)))
noisy_float32 = float32(clean_float64 + noise_scale * raw_noise)
```

Compute power over the whole retained clip, including silence. No silence detector or local per-frame SNR adjustment is used. At 20 dB, signal/noise power is 100; at 10 dB it is 10. Thus the 10 dB noise has sqrt(10), about 3.16 times, the RMS amplitude of the 20 dB noise for the same clip.

Measure realized SNR from the actual float32 output minus the clean input, not just from the requested scale. Abort if the difference from target exceeds 0.001 dB, rather than silently accepting an unrepresentable condition. Normalization by empirical noise RMS makes the finite-clip power ratio reproducible. This is a synthetic Gaussian-noise condition, not a recording of a specific room or device. Runtime versions are recorded; PCG64's integer-stream guarantee is not a promise that every distribution transform remains identical across future NumPy versions.

## Reduced bandwidth, with original duration

Use SciPy resample_poly at 16 -> 8 -> 16 kHz, with float64 on both passes, a Kaiser-window low-pass FIR (beta 5.0), and constant-zero boundary extension. A rate of 8 kHz has a 4 kHz Nyquist frequency; the filter has a transition band rather than a perfect brick-wall cutoff. This is not a complete telephone-channel simulation: no specified low-frequency cutoff, telephony quantizer or codec is added.

The downsampled length is ceil(N/2), so the upsampled result contains N samples for even N and N+1 for odd N. For odd N, remove the final extra sample. Cast back to float32, retaining exactly the original sample count at 16 kHz. There is no repeated padding, gain normalization or arbitrary duration change. Boundary effects are part of the documented zero-extension recipe. Controlled tests retain a 1 kHz tone within 1% RMS and suppress a 6 kHz tone below 0.3% of that reference RMS, excluding filter edges.

## Actual property audit

Every source matched the clean encoded and processed waveform fingerprints from [the prior recording audit](RECORDING_AUDIT.md). Each transform was calculated twice and compared sample-for-sample and field-for-field. All 4,200 outputs were finite mono float32 arrays with unchanged sample counts; all 4,200 replays were identical. Metadata and partition membership remain unchanged for 700 train and 700 dev recordings.

| Condition | Clips | Identical replays | Measured SNR range | Clips with peak > 1 | Maximum absolute peak |
| --- | --- | --- | --- | --- | --- |
| Gaussian noise, 20 dB SNR | 1400 | 1400 | 19.99999995 to 20.00000004 dB | 900 | 1.082217 |
| Gaussian noise, 10 dB SNR | 1400 | 1400 | 9.99999999 to 10.00000001 dB | 1125 | 1.260060 |
| 16 -> 8 -> 16 kHz | 1400 | 1400 | Not a noise-SNR condition | 655 | 1.148542 |

The bandwidth round trip removed one extra tail sample in each of 695 odd-length clips. These are length corrections after resampling, not discarded source recordings.

No clipping or peak normalization is applied. Noise can push samples outside [-1, 1], and FIR filtering can overshoot too. These remain valid in-memory float32 values, but exporting them directly to a bounded PCM format can introduce clipping. The audit records all above-unit samples and peaks. The next compression milestone must declare its headroom/clipping policy before the experiment is frozen; silently rescaling or clipping would add another transformation.

[The aggregate audit](PERTURBATION_AUDIT.json) contains exact settings, measured properties, input/code/runtime hashes and the checksum of local per-record evidence. Existing model-file checksums, the clean feature-index checksum and the prior threshold document are unchanged. These are waveform-property results, not degraded detection accuracy, recall or robustness conclusions.

## Reproduce

Prepare the [recording audit](RECORDING_AUDIT.md), then run from the repository root with the existing audio dependency group:

```bash
python -m voicefake audit-perturbations --manifest data/manifests/baseline-train-dev.csv --audio-root data/asvspoof2019/LA --audit-report docs/RECORDING_AUDIT_RESULTS.json --recordings artifacts/audits/day8/recordings.json --output-dir artifacts/perturbations/day9 --config configs/perturbations.json
python -m unittest discover -s tests -v
```

Use a fresh output directory. The runner validates the manifest and prior audit evidence, rejects evaluation membership before decoding, then verifies each loaded clean source before transforming it. A source/runtime mismatch or a failed property check prevents a completed output directory. It writes only report.json and transforms.json; the public report adds independent validation evidence. No new dependencies or audio acquisition are required.

All **94 tests pass**, including independent norm-based SNR measurement, paired noise levels, deterministic seeds/order, global-RNG isolation, immutable inputs, odd-length handling, spectral checks, unclipped peaks and rejection of changed source evidence or evaluation audio. The next stage adds compression and freezes the full choices before any final evaluation. Later paired robustness summaries will use unchanged clean-trained models and development-selected thresholds.

## References

- [SciPy resample_poly filtering and boundary semantics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- [NumPy PCG64 seeding and stream compatibility](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html)
