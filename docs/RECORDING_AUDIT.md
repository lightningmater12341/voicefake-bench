# Exact recording-content and source-membership audit

This milestone audits the existing train/development sample used by the models and operating thresholds. It does not select new recordings, refit a model or change a threshold. The three official key files are checked as metadata; evaluation audio is refused by this command.

## Check membership before decoding

The source keys must match the SHA-256 values pinned in [the original protocol audit](PROTOCOL_AUDIT.json). Parse all train/dev/eval keys and validate the complete source metadata before selecting recordings. This catches a speaker crossing partitions even when the offending entry is absent from the small audio sample.

Every selected path, label, partition, speaker ID and attack ID must exactly match its official record. Resolve local audio paths within the supplied dataset root. Manifest, feature-index membership, cache identities, feature-file checksums and vectors are checked before decoding. The audio recipe, feature-extractor source code and recorded runtime must still match those used to extract the features.

## Why three fingerprints?

| Level | What is hashed | What it catches |
| --- | --- | --- |
| Encoded file | SHA-256 of the file bytes | A copied file under a different path |
| Native decoded waveform | Original rate, frame count, channel count/order and float64 sample bytes | Exact decoded copies stored using different headers or lossless containers |
| Processed waveform | Existing mono/resample/center-crop recipe, output rate/shape and float32 sample bytes | Different source recordings that become identical inputs after preprocessing |

Waveform fingerprints use the schema waveform-v1: compact sorted JSON header, a newline, then C-order little-endian sample bytes. The header includes schema, sample rate, frame count, channel count and dtype. Positive and negative floating zero are canonicalized to the same silence. All other finite sample values remain unchanged; no tolerance or perceptual comparison is applied. Including rate and shape prevents identical flattened bytes with different audio interpretations from matching.

Native decoding retains all original frames and channels before processing. The bounded source-duration check runs before reading samples. The processed view uses the same process_samples function and configuration as feature extraction. Its existing bare float32 SHA-256 must match the feature cache's waveform hash, and its encoded source SHA-256 must match the feature-cache source identity. This establishes which model inputs were actually audited. Source files are hashed before and after decoding to reject an input changed during the scan.

## Duplicate policy

Group recordings separately by each fingerprint. Report within-partition groups, cross-partition groups and groups with contradictory labels, along with affected and redundant recording counts. A pair may appear at multiple fingerprint levels; do not sum those counts as distinct duplicate pairs.

Any duplicate group changes status to review_required and makes the CLI exit with status 1 after preserving the report and local group details. This includes same-label duplicates inside one partition, since repeated content can distort training weight or sample counts. The command never removes recordings, moves them between official partitions, silently changes labels, or edits the experiment. Review an actual finding and document any revised experiment before continuing; old measurements must remain attributable to their original sample. Input-validation failures exit with status 2 and do not create a completed audit directory.

## Reproduce

Use the same audio, original keys, manifest and features prepared by [the dataset](DATASET.md) and [baseline recipes](BASELINES.md). From the repository root:

```bash
python -m voicefake audit-recordings --manifest data/manifests/baseline-train-dev.csv --audio-root data/asvspoof2019/LA --train-protocol data/asvspoof2019/ASVspoof2019.LA.cm.train.trn.txt --dev-protocol data/asvspoof2019/ASVspoof2019.LA.cm.dev.trl.txt --eval-protocol data/asvspoof2019/ASVspoof2019.LA.cm.eval.trl.txt --protocol-reference docs/PROTOCOL_AUDIT.json --index artifacts/features/baseline-index.json --cache-dir artifacts/features/cache --audio-config configs/audio.json --output-dir artifacts/audits/day8
python -m unittest discover -s tests -v
```

Choose a fresh output directory. The ignored local run contains report.json, recordings.json with per-clip fingerprints, and duplicate_groups.json with group members. The public aggregate report records hashes of these local artifacts; it does not redistribute recordings or source keys. No new dependencies or audio acquisition are needed.

## Observed audit, 17 October 2026

All **1,400 recordings** passed: 700 official training clips and 700 development clips, with 100 genuine and 600 spoof clips in each partition. Every selected metadata record matched the pinned keys. The complete 121,461 source records were checked for speaker separation, including evaluation metadata; no evaluation recording was opened. The selected sample contains 40 speaker IDs; the complete source metadata contains 107.

| Fingerprint view | Duplicate groups | Cross-split groups | Within-split groups | Label-conflict groups |
| --- | --- | --- | --- | --- |
| Encoded files | 0 | 0 | 0 | 0 |
| Native decoded waveforms | 0 | 0 | 0 | 0 |
| Processed model-input waveforms | 0 | 0 | 0 | 0 |

The scan checked **86,733,549 encoded audio bytes**. No selected file needed resampling; one recording used the existing centre-crop rule. All 1,400 encoded sources and processed waveforms matched the feature-extraction evidence, and the feature-file checksums and vectors passed validation. No data, model setting, operating threshold or previous measurement was changed.

All **82 tests pass**. Controlled fixtures verify lossless re-encoding, channel mixing that creates identical processed inputs, within/cross-split duplicates, conflicting labels, duration guards, source-key pins, official membership, and rejection of changed audio or feature caches. A separate CLI check verifies that duplicate findings preserve local details and return exit status 1. A full second scan reproduces the entire aggregate report and both local artifact checksums, confirming identical per-record fingerprints. The audited manifest and feature-index hashes match the published metric provenance; existing model-file checksums are unchanged.

[The aggregate result](RECORDING_AUDIT_RESULTS.json) records counts, source identities, fingerprint conventions, runtime, code hashes and local-artifact checksums. It adds publication-time validation evidence to the deterministic runner report. Detailed per-record fingerprints remain local.

## Limits

A clean exact-hash audit applies only to the specified sample, decoder/runtime and preprocessing recipe. It does not establish that the entire corpus is free of overlap. Time-shifted, gain-scaled, lossy-compressed or otherwise similar recordings can evade exact hashes; near-duplicate or perceptual matching is not implemented. Speaker metadata separation is also not proof of content independence. Evaluation recording-content checks remain necessary after experiment freeze, before interpreting final results; evaluation audio is not decoded in this milestone.
