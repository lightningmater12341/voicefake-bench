# Dataset acquisition and manifest recipe

## Provenance and terms

This project uses **ASVspoof 2019 Logical Access (LA)**. The authoritative record is [University of Edinburgh DataShare](https://doi.org/10.7488/ds/2555), published by CSTR. Cite Yamagishi et al. (2019), *ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database*. For published experiments, also cite Todisco et al. (2019), *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*, Interspeech.

The dataset [README](https://datashare.ed.ac.uk/server/api/core/bitstreams/765597d6-633f-418a-96b0-39274a8d56c9/content) identifies the **Open Data Commons Attribution License (ODC-By) 1.0**. Retain attribution and the [supplied license](https://datashare.ed.ac.uk/server/api/core/bitstreams/60c7de6d-37d3-45c1-b52e-9f18b0338e47/content) when working with the database or derived metadata. ODC-By addresses database rights; its text distinguishes individual contents and other rights. This repository distributes code, documentation, and aggregate metadata audits, not speech recordings or the source keys.

Contains information from [ASVspoof 2019](https://doi.org/10.7488/ds/2555), made available under the [ODC Attribution License](https://opendatacommons.org/licenses/by/1.0/).

## Storage and acquisition

The official LA archive is **7,640,952,520 bytes (about 7.12 GiB)**. Its published MD5 is `30c98f11d8b2bc21f2c257bfd78bb5c5`. Download it from the dataset record; preserve README and license files. Before extracting, check space for both the archive and extracted audio, plus later feature caches. Subset selection limits later processing but does not reduce a normal full-archive download.

For the 10 October metadata check, only ZIP directory information, the three CM keys, and README/license metadata were fetched. Bounded HTTP ranges transferred 13,851,982 bytes from LA.zip; no audio was acquired. Individual extracted ZIP members passed Python's ZIP CRC checks. The published archive MD5 is recorded for future full downloads; it was **not verified against the entire archive** in this partial acquisition.

The importer consumes local protocol files. It does not implement network downloads. To reproduce using the standard full archive, extract LA.zip locally and use its official keys. Today’s source hashes and aggregate checks are in [PROTOCOL_AUDIT.json](PROTOCOL_AUDIT.json).

## Import the official keys

Use **CM** (countermeasure) keys, not the separate speaker-verification ASV protocols. Each record has five whitespace-separated fields: speaker ID, recording ID, a placeholder, spoof-method ID, and genuine/spoof label. The parser checks recording prefixes against the supplied partition and validates labels and attack IDs.

From the repository root, after `python -m pip install -e .`, run this as one line:

```bash
python -m voicefake import-protocols --train data/asvspoof2019/ASVspoof2019.LA.cm.train.trn.txt --dev data/asvspoof2019/ASVspoof2019.LA.cm.dev.trl.txt --eval data/asvspoof2019/ASVspoof2019.LA.cm.eval.trl.txt --output data/manifests/smoke.csv --per-stratum 100 --seed 42
```

Place the original keys at the paths above or supply their actual local paths. Audio paths in the generated manifest are relative to the extracted **LA directory**, for example `ASVspoof2019_LA_train/flac/LA_T_....flac`. Keeping protocols together locally does not change official train/dev/eval membership.

The command writes the CSV and an adjacent `smoke.csv.json` provenance file. It refuses to overwrite either existing output. Use a fresh filename for another experiment. `--all` includes all records in the supplied keys; it is mutually exclusive with `--per-stratum`.

```bash
python -m voicefake validate-manifest data/manifests/smoke.csv
```

Generated manifests, original keys, audio, and caches belong under ignored `data/` or `artifacts/`. Do not commit them. Only the aggregate audit is published here.

## Deterministic subset rule

Group records by original **partition, label, and spoof-method ID**. Genuine speech has method `-`. Rank each group by SHA-256 of compact JSON containing seed, partition, label, method, and relative audio path, then retain up to the cap. Sort selected records by partition and path for byte-stable CSV output.

The default seed is 42 and cap is 100. This produces **2,800 metadata entries**: 700 training, 700 development, and 1,400 evaluation. Changing input row order leaves the selected CSV unchanged; a larger cap retains the smaller-cap selections within each group. Input hashes, cap, seed, selection version, counts, and manifest hash are recorded.

The complete inputs are audited before sampling. This prevents a small sample from hiding a speaker overlap elsewhere in the source keys. Source keys remain local; the manifest validator does not load or hash audio yet.

## Observed counts, 10 October 2026

| Partition | Genuine in source | Spoof in source | Genuine selected | Spoof selected |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,580 | 22,800 | 100 | 600 |
| Development | 2,548 | 22,296 | 100 | 600 |
| Evaluation | 7,355 | 63,882 | 100 | 1,300 |
| Total | 12,483 | 108,978 | 300 | 2,500 |

All **121,461 source entries** passed metadata validation with no speakers shared between partitions. These are record counts, not model-performance results. The audit includes speaker IDs from both genuine and spoof records.

The smoke manifest SHA-256 is `239b0690f1a2afe608228efe588916f7d517afc58d0fada0c56998cbd5f98d02`.

## What this subset can establish

This is an initial pipeline subset, not a full challenge evaluation. Equal per-method caps alter class and attack prevalence and do not guarantee equal speaker representation. With only 100 genuine development records, a 1% false-positive target has one-record resolution. Choose and document larger development/evaluation sizes before reporting a useful operating-point benchmark. No model has been trained or scored in this milestone, and no audio-file existence or content-duplicate audit has been performed.
