#!/usr/bin/env bash
# Fine-tune ONLY the accent-similarity (acc_sim) LoRA adapter.
# Thin wrapper over pipeline-finetune.sh so all training params live in one file.
# Runs standalone (interactive) or from jobs/finetune-acc_sim.slurm.
set -euo pipefail
cd "$(dirname "$0")"
exec bash pipeline-finetune.sh acc_sim
