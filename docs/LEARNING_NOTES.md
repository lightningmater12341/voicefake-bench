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

## Day 7 — ranking quality and a useful decision boundary

**What changed:** We measured ROC-AUC and a documented interpolated EER from the existing Day 6 scores, then selected a threshold for each existing model using genuine development scores only. No models were refitted and no evaluation audio was touched. All 70 tests pass, and replay produces the identical metric report.

**What does ROC-AUC mean?** It measures how often a spoof receives a higher score than a genuine example, counting an equal-score pair as half a win. It looks across thresholds, so a high AUC does not guarantee good recall at the very low false-positive rate we care about. Random forest has AUC 0.918183 versus SVM's 0.902800, yet SVM detects more spoofs at our selected operating points.

**What does EER mean?** It is where the false-positive and false-negative rates meet along the ROC curve. Our implementation groups identical scores and linearly interpolates between adjacent empirical points. A virtual crossing can lie between achievable decisions, so we report the interpolated rate and its bracket rather than pretending it supplies a deployable threshold. Forest's development EER is 15.33%, SVM's is 18.00%, and logistic's is 21.00%, under this convention.

**How did we pick the actual threshold?** With 100 genuine development examples and a 1% target, we permit at most one false alarm. Sort their scores and put the threshold just above the second-highest score. Any scores tied at that boundary are excluded together. The selector never sees spoof scores. A saved rule classifies score >= threshold as spoof and can be applied later without recomputing its cutoff.

**Why does a tie matter?** Two genuine clips can have the same score. We cannot allow one and exclude the other by row order when the model gave them the same evidence. Keeping the entire boundary block excluded may use less than the allowed error budget. The constant majority reference illustrates this: every score is 1, so its selected threshold flags nothing and detects no spoofs.

**What happened at the selected point?** SVM flags one of 100 genuine clips and detects 401 of 600 spoof clips: 1% empirical FPR and 66.83% spoof recall. It misses 199 spoofs. At its default threshold it detected 471 spoofs but falsely flagged ten genuine clips. Reducing false alarms sacrifices detections. Forest detects 385 spoofs at one false alarm; logistic detects 235.

**Can I promise a 1% false-positive rate?** No. This is the rate on the small development sample used to select the threshold. Different speakers, methods and recording conditions can change it. Even 99 genuine clips would have an integer budget of zero at a 1% target. We must carry the saved threshold unchanged to the later frozen evaluation and report what actually happens there.

**Why check score provenance?** A metric is meaningless if scores and labels refer to different clips or row orders. The runner checks the source manifest and score-file hashes, label order, positive orientation and reproduction of the original default-threshold metrics. It can calculate today's report without loading audio or deserializing model binaries.

**Interview answer today:** "I separate global ranking quality from behavior at a chosen operating point. My forest has the best development AUC, but SVM detects more spoofs when each model is limited to one false alarm among 100 genuine development clips. Thresholds are selected from genuine development scores alone, ties stay together, and I do not claim that this small sample establishes the same rate in production."

**Try explaining:** Why might a higher AUC fail to give higher low-FPR recall? Why can an interpolated EER lack a single usable threshold? Why does one false alarm among 100 development clips fail to establish a population guarantee? What would go wrong if you recalculated the threshold on evaluation genuine scores?

## Day 8 — check whether recordings leak across partitions

**What changed:** We audited all 1,400 existing train/dev recordings at three levels: file bytes, native decoded audio and processed model-input waveforms. No exact duplicate groups were found at any level. Every selected record matched the pinned official metadata, and its audio still matched the source and processed waveform recorded during feature extraction. All 82 tests pass. A full replay reproduces the report and every per-record fingerprint. The audit does not change the models, thresholds or previous measurements.

**Why are different filenames insufficient?** Copying a recording and renaming it does not create an independent example. If a detector trains on that content and sees it again during model selection, the apparent generalization can be optimistic. A byte hash catches exact file copies; comparing only paths would miss them.

**Why also decode the audio?** WAV and FLAC containers can contain the same samples while their file bytes differ. The native waveform fingerprint includes the sample rate, frame count, channel count/order and canonical float64 samples. A controlled test writes identical PCM samples in WAV and FLAC and confirms that their file hashes differ while their decoded fingerprints match.

**Why hash after preprocessing too?** Different inputs can collapse to the same model input. For example, two stereo channels can average to an existing mono recording. Our fixture demonstrates this. Hashing the existing mono/resampled/cropped float32 waveform catches the collision. One real clip in this sample uses the established centre-crop rule; its processed hash matches the earlier feature evidence.

**Why include rate and shape?** The same sequence of sample bytes can mean a different duration at another sample rate, or a different arrangement of channels. Hashing the header together with canonical little-endian sample bytes prevents those interpretations from being conflated. Positive and negative floating zero are treated as the same silence.

**Why recheck all official metadata?** A tiny selected sample could omit the row that reveals a speaker overlap. The audit first verifies all three key-file hashes and validates all 121,461 source metadata records, then checks every selected path, label, speaker, attack and split against its official record. Evaluation keys are metadata only; no evaluation audio is opened.

**What if duplicates appear?** The command preserves group details and returns a review status, including for within-partition duplicates or conflicting labels. It never repairs the experiment by quietly dropping or relabeling clips. Any necessary change must be documented so that old measurements remain attributable to the data they actually used.

**What can I claim?** No exact content duplicates were found in these 1,400 clips under the recorded decoder and processing recipe. I cannot infer that the whole corpus has no overlap, or that shifted, gain-changed or compressed copies are absent. Exact hashes are not a perceptual similarity detector, and a clean audit does not itself improve model performance.

**Interview answer today:** "I checked more than speaker IDs and filenames. My audit compares encoded files, native decoded audio and the processed waveforms the models actually used. All 1,400 train/dev clips passed without exact duplicate groups, and I can link them to the pinned official keys and original feature inputs. I keep that claim limited to this sample and exact equality."

**Try explaining:** How can different file hashes hide identical audio? How can different stereo recordings become the same mono input? Why check complete source metadata before looking at a subset? Why must a duplicate finding preserve the original experiment history?

## Day 9 — change the audio in a controlled, repeatable way

**What changed:** We added Gaussian noise at 20 and 10 dB SNR and a separate 16 -> 8 -> 16 kHz bandwidth condition. All 4,200 transforms on the existing 1,400 train/dev clips preserved their sample counts, produced finite float32 output and replayed identically. All 94 tests pass. No models were refitted, thresholds changed or detector scores calculated today.

**What does SNR mean?** It compares signal power to noise power on a logarithmic scale. Here power is mean squared amplitude over the whole retained clip, including silence. At 20 dB, signal power is 100 times noise power; at 10 dB it is ten times. The lower-SNR noise has about 3.16 times the RMS amplitude. We scale a realized noise vector, then verify the ratio after casting the mixed output to float32.

**Why not one global random seed?** A global generator advances as recordings are processed, so changing traversal order can change a clip's noise. We derive a separate seed from the base seed and clean waveform fingerprint. The same clean content gets the same noise direction, independent of filename, label or processing order. The two SNR levels share that direction and change only its scale, which keeps their comparison controlled.

**Why round-trip through 8 kHz?** Downsampling with a low-pass filter removes high-frequency information; returning to 16 kHz restores the expected sample rate, not the removed detail. An 8 kHz rate has a 4 kHz Nyquist frequency. It is a limited bandwidth condition, not a complete telephone codec or network simulation.

**What happens to odd-length clips?** Downsampling N samples by two gives ceil(N/2), and upsampling returns one extra sample when N is odd. We remove that final extra sample so duration and source pairing stay exact. This happened in 695 of the actual clips. The source clip itself is never trimmed further or overwritten.

**Why record values above one?** Noise addition and FIR filtering can produce float32 peaks outside [-1, 1]. In this sample, 900 clips exceeded unit peak at 20 dB noise, 1,125 at 10 dB, and 655 after the bandwidth round trip. We preserve those values and record them. A bounded PCM encoder may clip them, so the compression milestone must specify headroom or clipping explicitly before freezing the experiment.

**Does this prove the detector is robust?** No. Today proves that the audio changes are reproducible and match their stated properties. We still need to extract features and compare predictions using unchanged clean-trained models and stored development thresholds. We must not retune thresholds on every degraded condition or silently add noisy training examples while claiming an unchanged-model comparison.

**How are the partitions protected?** The runner requires the prior clean recording audit, checks its evidence hashes and membership, and verifies each loaded source waveform. All transformed copies retain their source label and partition; none becomes an independent example in another partition. Evaluation audio is rejected before decoding.

**Interview answer today:** "I defined the degradation recipes before measuring detector performance. Noise is keyed to the clean waveform and verified at its requested SNR after float32 conversion. Bandwidth changes preserve the original sample count. I validated all 4,200 transforms, documented peaks that matter for later encoding, and kept the clean models and thresholds unchanged."

**Try explaining:** Why is 10 dB less clean than 20 dB? Why can a global random generator make results depend on file order? Why does upsampling fail to restore removed frequencies? Why must a compression experiment declare what happens to above-unit peaks?

## Day 10 — make compression explicit and freeze the experiment

**What changed:** The fixed 32 kbps mono MP3 condition passed replay and original-length checks on all 1,400 train/dev clips. All 105 tests pass. We sealed the original data selection, model artifacts, clean-development thresholds, condition recipes and runtimes before touching evaluation audio. No new detector scores were computed.

**Why does compression need a recipe?** Codec name alone is incomplete. Bitrate, encoder implementation/build, algorithm quality, sample format, gain, metadata and decoder delay handling can all affect samples. We pin these choices and hash the actual bundled FFmpeg binary. MP3 bytes and decoded samples repeat exactly under this recorded build.

**What happened to above-unit peaks?** For the MP3 branch, we multiply by 0.5 before encoding and undo that fixed gain after decoding. This is not per-clip peak normalization. Encoder inputs reached at most 0.5 in the actual sample; decoded output reached 1.136721 after inverse gain, with 115 above-unit samples retained as float32. Noise and bandwidth remain separate conditions. If the fixed encoder headroom is exhausted later, the experiment stops and reports the failure instead of changing gain or silently clipping.

**What did real audio reveal?** An initially unaligned 28,827-sample input decoded to 28,847 samples, although several tone probes had passed. We rejected that run. The final declared codec input always adds a known zero tail to reach a multiple of 576 samples, verifies that exact padded decoded length, then removes only our known extension. This is a fixed length-based boundary rule, not a search for an alignment that makes results look better. Regression tests cover the discovered case and impulse alignment.

**What exactly is frozen?** The primary Day 6 SVM, three comparison models, their file hashes, the Day 7 thresholds, official metadata selection, clean audio/MFCC settings, separate degradation conditions and runtime identities. The primary choice still follows the original development balanced-accuracy rule; we did not switch models after seeing degraded results. A verifier checks 36 local file bindings and runtime identities.

**What about evaluation data?** We created only a metadata manifest for the original 100-genuine/1,300-spoof evaluation subset, preserving official membership. Its audio is still unacquired and unprocessed. It needs a content audit before later scoring. A 100-genuine-clip subset still cannot establish a precise production false-positive rate.

**Why freeze now?** After viewing evaluation results, changing a threshold, model or condition would make evaluation part of model selection. The content/file hashes make accidental drift visible and the published Git history records what was committed before evaluation. A hash is not a digital signature or a substitute for honest reporting.

**What remains to build?** Paired development robustness summaries and speaker-bootstrap uncertainty, cost measurement, the one frozen evaluation report and the small demo. The uncertainty plan is declared now: 1,000 paired speaker-block draws, seed 42, 95% percentile intervals, fixed thresholds and explicit invalid-draw counts.

**Interview answer today:** "I froze the experiment before evaluation: the same model files, clean-development thresholds and preprocessing must apply to every condition. I also tested the actual codec path, found a boundary-length discrepancy, and replaced guesswork with a declared frame-alignment rule. My verifier detects changed inputs, and my report distinguishes codec validation from detector performance."

**Try explaining:** Why can two implementations of MP3 define different experiments? Why is a fixed headroom gain different from peak normalization? What makes a documented padding rule different from silently repairing output lengths? Which choices become invalid if changed after viewing evaluation?

## Day 11 — measure robustness without moving the goalposts

**What changed:** We ran all four fixed models on the same 700 development recordings under clean, 20/10 dB noise, bandwidth reduction and MP3 conditions. We added 1,000 paired speaker-block bootstrap draws across 20 speakers, with 95% percentile intervals and explicit valid/invalid counts. Every draw was valid. All 115 tests pass; the 36 frozen bindings still verify. Evaluation audio remains untouched.

**What actually failed?** The primary SVM falsely flags 1/100 genuine clean clips, 11/100 after MP3 and 98/100 at 20 dB noise. At 10 dB and under bandwidth reduction it calls all 700 recordings spoof. This achieves 100% spoof recall while also flagging every genuine voice. Logistic fails differently: it calls every noisy clip genuine and misses all 600 spoofs. Low false alarms alone and high recall alone are both incomplete measures.

**Can AUC still look reasonable?** Yes. Bandwidth SVM AUC is 0.862983, yet every score lies above its original decision threshold. Scores can preserve useful ordering while shifting relative to the operating point. We report both ranking metrics and behavior at the unchanged threshold. We do not repair the result by choosing a new threshold for each condition.

**What does pairing reveal?** MP3 SVM detects 439/600 spoofs versus 401/600 clean. That net gain is 61 newly detected spoofs minus 23 previously detected spoofs now missed. Ten new genuine false alarms accompany it. A06 recall falls from 6/100 to 2/100 even though overall recall rises. Preserving source identity and attack counts exposes changes hidden by the aggregate rate.

**Why resample speakers?** Clips from one voice may share properties. Sampling 700 clips independently would ignore that grouping. We sample 20 speaker blocks with replacement, include every clip in each sampled block and retain duplicate-speaker multiplicity. All models and conditions use the same draw. Rates remain weighted by clip counts, not an equal average of per-speaker rates.

**What does a paired interval mean?** For each speaker draw, compute degraded minus clean, then take the declared percentiles of those differences. MP3 SVM FPR rises by 10 percentage points, with a paired 95% interval of [4.30, 16.81] points. Recall rises by 6.33 points, interval [1.75, 11.28]. Those are conditional development diagnostics, not population guarantees or an invitation to retune.

**Why are some intervals zero-width?** Every observed clip in that class receives the same fixed decision, so resampling existing blocks cannot create a different rate. An observed 100% rate and [100%, 100%] interval do not establish exact population certainty. Few speakers, 100 genuine clips, shared attack systems and model/threshold selection limit these intervals. We never refit or reselect inside a draw, so selection uncertainty is not covered.

**How were results checked?** All clean vectors and scores match their original evidence; all transformed waveforms match pre-freeze fingerprints. Predictions replay identically. An independent scikit-learn check reproduces all 20 AUCs, and a separate weighted-speaker calculation reproduces 80 rate/paired-change interval sets. Confusion matrices and per-attack counts match the actual saved score arrays, which remain local.

**Interview answer today:** "The interesting result is that my clean threshold does not survive degraded audio. SVM can detect every fake while falsely rejecting every genuine recording, and a respectable AUC can hide that failure. I kept the model and threshold fixed, compared the same recordings, and used shared speaker-block resampling to describe uncertainty. These are development diagnostics; the final evaluation is still untouched."

**Try explaining:** Why is 100% spoof recall insufficient? How can a useful ranking have a bad fixed threshold? Why must duplicated bootstrap speakers retain all their clips? Why is a paired interval different from subtracting two interval endpoints? Which uncertainty does a post-selection development bootstrap miss?
