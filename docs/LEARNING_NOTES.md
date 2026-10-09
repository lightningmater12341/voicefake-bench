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
