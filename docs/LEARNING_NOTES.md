# Learning notes

## Day 1 — what the benchmark is actually testing

**The problem:** Generated speech can resemble genuine speech. A detector may learn patterns associated with the examples it trained on, but those patterns might disappear with another generation method or compression.

**The project:** Compare several detectors under the same dataset and preprocessing rules, then measure how their behavior changes under predefined audio degradation.

**Why speaker separation matters:** If the same person appears during training and testing, a model may exploit speaker-specific patterns. Official separate-speaker partitions reduce that source of optimistic evaluation.

**Why there are three partitions:** Training fits the model. Development chooses settings and a decision threshold. Evaluation measures the frozen choices. Looking at evaluation to improve the model turns it into another development set.

**Today's code:** A manifest is a CSV inventory of audio and metadata. The validator rejects invalid entries and speakers crossing partitions before audio processing begins. It checks metadata only; duplicate-audio hashing comes later.

**A precise answer today:** "I've defined the protocol and implemented metadata checks. The models and performance experiments are the next milestones." There is no accuracy or performance result to quote yet.

**Try explaining:** Why might a detector with high overall accuracy still miss a new voice generator? Why should a compressed copy of a training clip stay in the training partition?

Future milestones will append their own explanations, actual observations, and likely interview questions.

## Day 2 — reproducible data comes before models

**What changed:** The importer reads the official five-field label files, preserves train/dev/eval membership, and builds a small metadata manifest with a saved recipe and hashes. The complete 121,461 input entries passed the metadata checks; the first smoke subset has 2,800 entries. No audio has been loaded and no models have been trained yet.

**Why not select the first 100 rows?** File order may group particular speakers or methods. Hash-based ranking selects records consistently without relying on that order. The same seed and inputs produce the same CSV, even when the input rows are reordered.

**Why record hashes?** A recipe such as "seed 42, cap 100" is incomplete if the source files change. Source hashes identify the exact input bytes; the manifest hash identifies the exact output bytes. They establish reproducibility, not that a detector performs well.

**What is a stratum?** Here it is a partition/label/method group. A separate cap for each spoof method keeps a small pipeline sample from losing a method entirely when every source group is nonempty. It also changes the dataset's original proportions, so results must say which sample was used.

**Why audit before sampling?** If a bad speaker overlap exists in the full input, a tiny subset might happen to leave out the offending rows. Checking the complete source keys first prevents that accidental pass.

**What remains unchecked?** Audio availability and decoded properties come next. Different filenames may still hide duplicate audio; content hashing is a later milestone. A 100-genuine-clip development subset is too small to make a strong 1% false-positive-rate claim.

**Interview answer today:** "I can reproduce my dataset selection exactly and show which source files produced it. I preserve official splits and check leakage before sampling. The current subset is for pipeline development; performance experiments are still ahead."

**Try explaining:** Why does changing the cap affect class prevalence? What is the difference between a source-file hash and an audio-content hash?

## Day 3 — make audio processing explicit

**What changed:** The loader now reads local recordings, averages channels to mono, resamples when needed, and applies a documented duration rule. The same processing applies to both labels. All 14 official train/dev smoke clips loaded successfully, repeated decoding was identical, and the full test suite passes 30 tests. No features or models have been built yet.

**What does 16 kHz mean?** The waveform contains 16,000 time samples each second. At this rate, its Nyquist frequency is 8 kHz. A consistent sample rate makes later feature windows correspond to the same duration and frequency range for every recording.

**Why filtering when downsampling?** Simply dropping samples can fold high-frequency energy into lower frequencies, called aliasing. The polyphase FIR resampler applies low-pass filtering as part of sample-rate conversion. A test checks that a 12 kHz tone is suppressed when reducing 48 kHz audio to 16 kHz.

**Why preserve silence and volume?** Trimming or normalizing is another modeling choice that could remove useful information or create shortcuts. This first recipe retains original silent segments and amplitude, then documents the choice. It does not claim those choices are universally best. Completely zero-energy signals are rejected because they cannot provide a useful waveform.

**Why not pad every clip?** The planned summary features can handle different clip durations. Padding would add artificial silence, and repeated padding could add artificial periodic patterns. Short clips retain their actual length; recordings longer than 10 seconds are center-cropped deterministically. Clips shorter than 0.25 seconds fail explicitly instead of being silently dropped.

**What did real clips test?** Actual dataset decoding and repeatability. Every selected clip was already mono at 16 kHz and shorter than 10 seconds, so crop and resampling behavior was tested with controlled signal fixtures instead. A small smoke check is not proof that every corpus file is valid.

**Interview answer today:** "I made preprocessing reproducible before extracting features. I record the source properties, resampling and crop decisions, library versions, and waveform fingerprints. The current check establishes that the loader works; it says nothing yet about fake-voice detection accuracy."

**Try explaining:** What happens to a 12 kHz tone if you downsample without filtering? How could padding or silence trimming affect a detector? Why should waveform hashes be interpreted alongside software versions?

## Day 4 — turn sound into comparable numbers

**What changed:** Each processed clip now becomes 40 numbers: 20 MFCC means and 20 standard deviations. The 14 official smoke clips produced finite vectors, and a second pass reused every cache entry with identical results. All 42 tests pass; there is still no trained detector.

**What is an MFCC?** We break speech into short frames, measure frequency-band energy on a mel scale, take a logarithm, and use a cosine transform to summarize the shape of that energy distribution. These are fixed signal-processing calculations. A later classifier learns which patterns are associated with genuine or spoofed examples.

**Why mean and standard deviation?** Different clips have different numbers of frames. Pooling each coefficient gives a fixed-size vector: its typical value and how much it varies. This is inexpensive and explainable, but pooling loses frame order and detailed timing information.

**Why specify every setting?** "MFCC features" is not a complete recipe. Window size, frequency scale, energy floor, DCT normalization, retained coefficients, and pooling all affect the values. Our configuration and processing notes make those choices reviewable.

**Why retain c0?** It contains overall log-energy-scale information. A mathematical test confirms that multiplying amplitude by two shifts c0 by the expected amount while the other coefficients stay unchanged when the energy floor is inactive. That also highlights a limitation: a model may use volume-related information, so later failure analysis matters.

**Why cache by content?** A filename can stay the same while its audio changes. Source hashes, processing settings, source code, and runtime versions identify the calculation. A matching validated cache entry avoids another decode and feature pass; a changed source or recipe cannot silently reuse stale numbers.

**Where could leakage happen next?** Computing fixed per-clip means is not fitting a dataset-wide scaler. Standardizing feature columns across all partitions would be a learned preprocessing mistake. The later model pipeline must learn its scaling on training data only. Labels and speaker/method IDs remain metadata, not numeric predictors.

**Interview answer today:** "I can turn variable-length speech into a reproducible 40-dimensional representation and show that my caches reuse only matching inputs and settings. I have verified the feature pipeline, and the next step is a baseline on a larger training/development sample."

**Try explaining:** What information does mean/std pooling discard? Why can the same filename need a fresh cache entry? How is a fixed per-clip summary different from fitting a scaler on the whole dataset?

## Day 5 — measure a baseline and read its mistakes

**What changed:** The local dataset expanded to 700 training and 700 development clips, with 100 genuine and 600 spoof clips in each partition. The first models are a majority-class reference and logistic regression on the same 40 features. Repeated fits and restored-model predictions match; all 51 tests pass.

**What does the majority reference teach us?** Calling every voice fake gets 85.71% raw accuracy because 600 of 700 development clips are spoofed. It also falsely flags every genuine voice. Balanced accuracy gives each class equal weight, exposing this reference's 50% result.

**What does logistic regression learn?** It learns weights for the feature columns and an intercept, producing a score through the logistic function. StandardScaler first puts feature columns on comparable scales. Both the scaling statistics and model weights are fitted using training data only.

**Why class weighting?** There are six times as many spoof examples as genuine examples. The balanced setting uses training counts to give the two classes equal total weight in the training objective. That choice changes the tradeoff; it does not automatically maximize raw accuracy or calibrate the scores.

**What happened on development?** Logistic regression achieved 78.92% balanced accuracy, recognizing 72 of 100 genuine clips and 515 of 600 spoof clips. It falsely flagged 28 genuine clips and missed 85 spoof clips at the predefined 0.5 threshold. Its raw accuracy was 83.86%, below the majority reference's 85.71%. Reporting the confusion matrix makes that tradeoff visible.

**Why leave these imperfect results in the project?** A benchmark measures a starting point and its failure modes. Selecting or hiding results to make the first model look impressive would defeat that purpose. Same-data comparisons and later operating-point analysis can establish what actually improves.

**Why keep development separate?** Today's model settings were chosen before inspecting its scores. Future model selection may use development results, which means those results remain provisional. The held-out evaluation partition must stay separate from those choices.

**Interview answer today:** "My first baseline is reproducible, and its confusion matrix shows a meaningful weakness: it falsely flags 28% of genuine development clips at the default threshold. I compare against a majority reference to show why raw accuracy is misleading under imbalance. The next steps are same-data model comparisons and a carefully selected operating point."

**Try explaining:** Why can a model with lower raw accuracy be more useful here? What does class weighting change? Why do we need a final partition after using development results to choose models?

## Day 6 — compare models and explain the tradeoff

**What changed:** We tried six RBF SVM settings and four random-forest settings using the exact Day 5 training/development clips and MFCC features. Every candidate fits training data only. The small grid, decision rules and selection metric were fixed before scoring. All 58 tests pass; the real-data search repeats exactly, and restored models reproduce the saved development scores.

**What does an RBF SVM add?** Logistic regression draws a linear boundary in standardized feature space. The RBF kernel lets an SVM express a curved boundary through similarity to training examples. C controls the penalty for training errors: a smaller C regularizes more strongly. Gamma controls how local the similarity is. Scaling matters because distances depend on feature units; learning the scaler from development data would leak information into fitting.

**What does a random forest add?** It averages many decision trees trained using bootstrap samples and random feature choices. Trees split individual columns and do not need standardization. Depth and minimum leaf size control how detailed the splits become. A seed makes the random choices repeatable; it does not guarantee generalization.

**How did we choose settings?** Within each family we selected the highest development balanced accuracy, keeping the first listed setting on an exact tie. The selected SVM uses C=0.1 and gamma=scale; the selected forest uses depth 12 and minimum leaf size 3. Logistic kept its earlier fixed settings, so this is a bounded benchmark comparison rather than an exhaustive or equally tuned contest.

**What happened?** SVM reached 84.25% development balanced accuracy, compared with logistic's 78.92% and forest's 76.83%. SVM falsely flagged 10 of 100 genuine clips and detected 471 of 600 spoofs. Logistic falsely flagged 28 genuine clips but detected more spoofs: 515 of 600. The model with the best balanced accuracy did not win on every measure.

**Why not choose the forest's 88.14% raw accuracy?** It falsely flagged 39% of genuine development voices. There are six times as many spoof clips, so raw accuracy rewards the majority class heavily. Its 98.33% training balanced accuracy also drops to 76.83% on development, which is consistent with overfitting this sample.

**Are the scores interchangeable?** No. SVM emits a signed margin, with zero as today's decision boundary. Logistic and forest emit spoof probabilities, with 0.5 as today's boundary. Higher means more evidence of spoof for all three, but neither their magnitudes nor their confidence interpretations are interchangeable. We have not calibrated probabilities or chosen a low-false-positive threshold yet.

**Why can't I call this the final result?** We used development scores to choose parameters and a model, so development is part of the selection process. It contains known attack methods and only 100 genuine examples. We must keep the evaluation partition untouched until settings and thresholds are frozen, then measure generalization there.

**Interview answer today:** "I compared three classical classifiers on identical speech features and official speaker-separated data. A small predefined search selected SVM by development balanced accuracy, but its confusion matrix shows the cost: fewer false alarms and more missed spoofs than logistic regression at their default thresholds. I record every candidate and preserve evaluation data for a later frozen experiment."

**Try explaining:** Why does SVM need scaling while trees do not? Why is a margin not a probability? How can the best raw accuracy hide worse false alarms? Why do parameter selection results need a separate final evaluation?
