# Working on VoiceFake Bench

This project is intentionally built and published in small daily increments at the user's request.

- Read README.md, docs/ROADMAP.md, and the local build-state file before proceeding.
- Implement only the next unfinished milestone per daily run. A milestone may take several runs; do not race ahead or upload the entire planned project.
- Publish at most one project-progress commit per Asia/Kolkata calendar date. Check both local state and remote history before publishing, including runs retried after an interrupted push. Do not backdate commits, create empty activity commits, force-push, or catch up with a bulk upload.
- Validate the relevant component and update its roadmap status and learning notes in the same commit. Treat fixtures as fixtures and never report fabricated model results.
- Keep raw audio, model binaries, caches, local absolute paths, credentials, and build-state files outside tracked Git files. Explicitly stage project files; do not stage the surrounding workspace.
- Preserve official speaker-separated partitions. Use training for fitting, development for selection, and frozen evaluation for reporting. Never tune based on evaluation results.
- Read data license and storage requirements before acquiring audio. A pipeline subset and the full challenge are different experiments; report which was used.
- Existing user authorization includes ordinary commits and uploads to this project's GitHub repository. Ask only when missing information or an actual external restriction prevents progress.
- Once the roadmap is finished, update the daily automation to PAUSED and report completion. Do not keep adding unsolicited work.
