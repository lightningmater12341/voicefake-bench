# Local frozen-model demo

The final component is a terminal command for one local recording. It uses the original primary SVM and frozen threshold. It prints JSON; it does not start a server, send audio anywhere, save predictions, retrain, or update the benchmark.

## Run on the existing project machine

From the repository root, with the project environment active:

```powershell
python -m voicefake verify-freeze
python -m voicefake demo-audio data/asvspoof2019/LA/ASVspoof2019_LA_dev/flac/LA_D_1086391.flac
python -m voicefake demo-audio data/asvspoof2019/LA/ASVspoof2019_LA_dev/flac/LA_D_1022459.flac
python -m voicefake demo-audio "path/to/your-recording.wav"
```

On the original Windows checkout, use `.venv/Scripts/python.exe` instead of `python` if the environment is not active. For a different working directory, pass `--root path/to/voicefake-bench` and an appropriate recording path.

The first two commands use existing **development** recordings, not evaluation examples. The genuine example produces margin -0.7201291994029779 and `not_flagged_as_spoof`; the spoof example produces margin 1.1982569889868373 and `flagged_as_spoof` in the pinned environment. They demonstrate plumbing, not accuracy. Validation selected the first two development entries of each label in the original feature-index order, without choosing examples based on predictions.

## What the output means

The JSON includes the signed SVM margin, frozen decision rule, model/freeze hashes, input/processed hashes, feature hash, and audio-processing details. Larger margins point toward spoof. A margin **at least 0.262431783372518** flags spoof; equality flags spoof. A signed margin is **not a calibrated probability**. There is no confidence percentage.

`not_flagged_as_spoof` means the score is below that threshold. It does not verify genuine speech. A flag is also not proof of how a recording was made. The measured [evaluation failures](EVALUATION.md) apply: on the declared clean subset the SVM missed 484 of 1,300 spoofs, and MP3 caused 22 false alarms among 100 genuine clips. Arbitrary microphone or current-generator audio is outside the measured distribution.

The command retains the [audio rules](AUDIO.md): average channels, resample to 16 kHz, accept source duration 0.25-120 seconds, centre-crop to at most 10 seconds, and keep amplitude unchanged. It rejects zero energy, invalid samples and invalid duration. It extracts the same 40 MFCC statistics, uses the saved training-fitted scaler and SVM, and applies the saved threshold. The demo processes the supplied recording as-is; it does not generate benchmark perturbations or diagnose speech quality.

## Required local artifacts and failure behavior

This is a demo of the **completed local experiment**. A fresh GitHub clone includes source, configuration and aggregate evidence; it deliberately excludes audio, feature caches and joblib model files. Installing dependencies alone cannot make this frozen demo run. The frozen environment and exact local artifacts described in [dataset setup](DATASET.md), [model comparison](MODEL_COMPARISON.md) and [freeze documentation](FROZEN_EXPERIMENT.md) must be available. Runtime drift or any of the 36 changed/missing frozen bindings causes failure. Original validation used Python 3.14.7 and the versions recorded in the freeze, although the package's general utilities support Python 3.10+.

The demo verifies the freeze before decoding or loading a model. It deserializes only the trusted repository-relative, hash-verified SVM artifact. Never substitute an untrusted joblib file.

It also requires the completed evaluation audit and its local `artifacts/evaluation-audit/day13/recordings.json`, checked against the public audit hash and frozen manifest. This existing metadata lets the demo reject the selected held-out recordings without reopening their audio. Original evaluation paths and encoded copies fail before decoding; exact matches after canonical processing fail before model loading/inference. Those checks cover exact selected evaluation content, not perceptual near-duplicates or modified copies with different processed samples. They are a guard against accidental reuse, not a general dataset-recognition system. Missing or changed guard evidence causes failure; there is no bypass option.

Input errors exit with status 2 and a reason. No model or threshold option is exposed. Troubleshoot missing original artifacts or environment drift; do not reseal the experiment, choose another model, rerun evaluation, or adjust the threshold to make an example look better.

## Validation

[DEMO_VALIDATION.json](DEMO_VALIDATION.json) records four real development checks: four original feature hashes and decisions matched, four margins matched saved scores within 1e-12, repeated outputs matched, and CLI output matched library output. The evaluation CLI test exited 2 without decoding that recording. All 32 saved evaluation evidence files were rehashed unchanged. Per-recording demo checks stay in ignored local artifacts.

Ten new boundary tests cover threshold equality, unchanged files, freeze-first failure, evaluation path/encoded/processed-copy refusal, invalid audio, stereo resampling/cropping, non-finite scores, primary-model enforcement and guard provenance. Synthetic signals and fixture models in tests are not detection evidence. The full suite passes 146 tests:

```bash
python -m unittest discover -s tests -v
```
