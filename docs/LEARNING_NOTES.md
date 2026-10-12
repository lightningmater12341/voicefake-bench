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
