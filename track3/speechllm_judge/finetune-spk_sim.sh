#!/usr/bin/env bash
# Fine-tune ONLY the speaker-similarity (spk_sim) LoRA adapter.
# Thin wrapper over pipeline-finetune.sh so all training params live in one file.
# Runs standalone (interactive) or from jobs/finetune-spk_sim.slurm.
set -euo pipefail
cd "$(dirname "$0")"
exec bash pipeline-finetune.sh spk_sim
