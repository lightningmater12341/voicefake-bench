# Daily build roadmap

Each milestone is one small, meaningful daily increment. A milestone can take more than one day if data access, tests, or experiments need longer. Do not compress missed days into a bulk upload or create empty commits for activity.

| Step | Increment | Evidence before publishing | Status |
| --- | --- | --- | --- |
| 1 | Define benchmark, create Python package, validate manifest metadata | Leakage/metadata tests and CLI check | Complete |
| 2 | Document dataset terms and storage; import official LA protocols and deterministic subset manifests | Official metadata audited: 121,461 records; reproducible 2,800-entry subset; 18 tests pass | Complete |
| 3 | Load audio and establish sample-rate/duration/silence rules | 14 official train/dev clips decode repeatably; processing recipe and runtime recorded; 30 tests pass | Complete |
| 4 | Extract and cache MFCC summary features | 14 × 40 finite feature matrix; identical cache replay; no fitted preprocessing; 42 tests pass | Complete |
| 5 | Add majority-class and logistic-regression baselines | 700 train + 700 dev clips; actual majority/logistic metrics; identical repeat scores; 51 tests pass | Complete |
| 6 | Add SVM and random forest with a bounded search | Same 700 train + 700 dev and 40 features; ten candidates; repeated metrics/scores and restored predictions match; 58 tests pass | Complete |
| 7 | Implement EER, ROC-AUC, and development-selected operating threshold | Tie-grouped ROC/AUC and interpolated EER; genuine-only integer FPR budget; verified saved-score replay; 70 tests pass | Complete |
| 8 | Audit recording hashes and data leakage | 1,400 train/dev clips; zero encoded/native/processed duplicate groups; pinned full-key membership and feature inputs verified; 82 tests pass | Complete |
| 9 | Add fixed noise and bandwidth perturbations | Reproducibility and audio-property checks | Planned |
| 10 | Add compression condition; freeze model/threshold/configuration choices | Codec/version record and frozen experiment specification | Planned |
| 11 | Add paired robustness summaries and speaker-bootstrap uncertainty | Development checks; documented resampling assumptions | Planned |
| 12 | Measure feature-extraction and model inference costs | Hardware, timing procedure, and actual runtime logs | Planned |
| 13 | Run frozen evaluation once and write failure analysis | Actual clean/degraded scores, per-attack counts, reproducible report | Planned |
| 14 | Add a small local demo and finish interview walkthrough | Demo tested; explanation matches implemented work and limitations | Planned |

The initial scope uses classical models on a laptop. A neural comparison or a newer external corpus is a later extension, justified by the first benchmark's findings.
