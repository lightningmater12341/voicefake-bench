# Daily build roadmap

Each milestone is one small, meaningful daily increment. A milestone can take more than one day if data access, tests, or experiments need longer. Do not compress missed days into a bulk upload or create empty commits for activity.

| Step | Increment | Evidence before publishing | Status |
| --- | --- | --- | --- |
| 1 | Define benchmark, create Python package, validate manifest metadata | Leakage/metadata tests and CLI check | Complete |
| 2 | Document dataset terms and storage; import official LA protocols and deterministic subset manifests | Source/license record, counts, split audit; no raw audio in Git | Planned |
| 3 | Load audio and establish sample-rate/duration/silence rules | Real audio smoke check and recorded processing settings | Planned |
| 4 | Extract and cache MFCC summary features | Shapes, finite values, training-only preprocessing | Planned |
| 5 | Add majority-class and logistic-regression baselines | Reproducible training and development metrics | Planned |
| 6 | Add SVM and random forest with a bounded search | Same data/features; comparable development outputs | Planned |
| 7 | Implement EER, ROC-AUC, and development-selected operating threshold | Edge-case metric checks; positive label and tie policy | Planned |
| 8 | Audit recording hashes and data leakage | Duplicate-content report and source-partition checks | Planned |
| 9 | Add fixed noise and bandwidth perturbations | Reproducibility and audio-property checks | Planned |
| 10 | Add compression condition; freeze model/threshold/configuration choices | Codec/version record and frozen experiment specification | Planned |
| 11 | Add paired robustness summaries and speaker-bootstrap uncertainty | Development checks; documented resampling assumptions | Planned |
| 12 | Measure feature-extraction and model inference costs | Hardware, timing procedure, and actual runtime logs | Planned |
| 13 | Run frozen evaluation once and write failure analysis | Actual clean/degraded scores, per-attack counts, reproducible report | Planned |
| 14 | Add a small local demo and finish interview walkthrough | Demo tested; explanation matches implemented work and limitations | Planned |

The initial scope uses classical models on a laptop. A neural comparison or a newer external corpus is a later extension, justified by the first benchmark's findings.
