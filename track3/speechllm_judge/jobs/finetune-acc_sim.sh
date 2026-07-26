#!/usr/bin/env bash
#SBATCH --job-name=vmc-t3-ft-acc
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --output=/network/scratch/g/guzmand/Repositories/VoiceMOS-Challenge-2026/track3/speechllm_judge/logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

START_TIME=$(date +%s)
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

##################################################################
# Environment
##################################################################
module load miniconda/3
module load gcc/9.3.0
module load cudatoolkit/12.6      # sets CUDA_HOME (deepspeed import needs it)

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled

conda activate speecheval

echo "NVCC version:"; nvcc --version
echo "NVIDIA SMI:";   nvidia-smi

##################################################################
# Fine-tune the ACCENT-similarity adapter (single GPU)
##################################################################
cd /network/scratch/g/guzmand/Repositories/VoiceMOS-Challenge-2026/track3/speechllm_judge
bash finetune-acc_sim.sh

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
printf "Job %s finished on %s at %s\n" "$SLURM_JOB_ID" "$(hostname)" "$(date)"
printf "Total duration: %dh %dm %ds\n" $((DURATION/3600)) $(((DURATION%3600)/60)) $((DURATION%60))
