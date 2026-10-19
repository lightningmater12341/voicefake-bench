# Fixed MP3 condition

Validated on 19 October 2026, before any evaluation audio was acquired or scored. This condition starts separately from each clean processed waveform; it is not combined with noise or bandwidth changes. Existing clean-trained models and clean-development thresholds are unchanged.

## Exact encoding recipe

Use mono 16 kHz float32 input, FFmpeg's libmp3lame encoder at **32 kbps CBR**, algorithm quality 2, bit reservoir enabled, ABR disabled and one worker. The optional compression dependency is imageio-ffmpeg 0.6.0. Its bundled binary is FFmpeg 7.1 essentials; [the audit](COMPRESSION_AUDIT.json) records the exact version, binary SHA-256 and encoder-help hash. No executable is redistributed by this repository.

Multiply samples by the fixed gain **0.5** before encoding. Abort if the scaled peak reaches or exceeds 1; do not adapt gain to the clip, silently clip or skip it. Feed float32 PCM through a pipe, so no bounded integer-PCM conversion occurs. Decode using FFmpeg mp3float to float32, then apply the inverse gain **2.0**. Retain any decoded peaks above one in float32. These two fixed gains are part of the compression condition; no peak normalization is performed. Noise and bandwidth conditions remain as previously defined and are not sent through this codec.

## Explicit end handling

MP3 uses encoder delay and end padding. Write the encoded stream to a seekable temporary file so its Xing/LAME metadata is retained; strip ID3 tags and unrelated metadata and enable bit-exact output flags. The temporary MP3 is removed after decoding.

The initial unaligned-input probe passed several lengths but the real-data audit found a 28,827-sample clip decoding to 28,847 samples. A second decoder showed the same discrepancy. No result from that failed run was accepted. The declared recipe now zero-extends **every codec input to a multiple of 576 samples**, the MPEG-2 Layer III frame size at 16 kHz. This rule is based only on length, never on labels, signal content or detector scores.

Verify that gapless decoding returns exactly this padded input length. Any discrepancy aborts the condition; there is no alignment search or length-repair fallback. Only after that check, remove the precisely known extension we added and apply inverse gain. The waveform seen by feature extraction has exactly its original sample count. This is an explicit zero-boundary codec recipe, not a claim that the encoder handles every unaligned input perfectly. Regression tests cover the discovered boundary length and verify impulse alignment.

## Actual train/dev validation

| Check | Measured result |
| --- | --- |
| Existing official train/dev clips | 1,400 (700 train, 700 dev) |
| Exact original output sample counts | 1,400 |
| Identical repeated MP3 bytes and decoded samples | 1,400 |
| Maximum scaled encoder-input peak | 0.5 |
| Maximum output peak after inverse gain | 1.136721 |
| Output samples above unit magnitude, retained | 115 |
| Total declared zero-tail samples across clips | 403,775 |

All source files and clean processed-waveform hashes matched the prior recording audit. Each clip was encoded and decoded twice. Per-record codec/decoded fingerprints and known padding counts remain local; only [the aggregate report](COMPRESSION_AUDIT.json) is published. No model was refitted, threshold changed, detector score computed or evaluation audio processed. No waveform or MP3 files are retained by the audit.

All **105 tests pass**, including actual codec replays, odd and boundary lengths, source immutability, nonzero codec distortion, fixed headroom, unclipped float output, impulse alignment, evaluation-membership rejection and frozen-input drift. These are codec-property checks, not evidence that a detector is robust to compression.

## Reproduce

Prepare the previous [recording audit](RECORDING_AUDIT.md), then run from the repository root:

```bash
python -m pip install -e ".[audio,models,compression]"
python -m voicefake audit-compression --manifest data/manifests/baseline-train-dev.csv --audio-root data/asvspoof2019/LA --recording-audit docs/RECORDING_AUDIT_RESULTS.json --recordings artifacts/audits/day8/recordings.json --output-dir artifacts/compression/day10 --config configs/compression.json
python -m unittest discover -s tests -v
```

Choose a fresh output directory. The original [frozen experiment](FROZEN_EXPERIMENT.md) requires its recorded runtime/binary identities, not merely any package allowed by the optional dependency range. A different codec build defines a different reproducibility context and must not silently replace the frozen one.

## References

- [FFmpeg libmp3lame options](https://ffmpeg.org/ffmpeg-codecs.html#libmp3lame)
- [FFmpeg MP3 muxer and Xing/LAME metadata](https://ffmpeg.org/ffmpeg-formats.html#mp3)
- [imageio-ffmpeg packaging and executable discovery](https://github.com/imageio/imageio-ffmpeg)
