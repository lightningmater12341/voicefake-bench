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
