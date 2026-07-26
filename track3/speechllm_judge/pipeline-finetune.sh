# Vanilla LoRA SFT of SQ-LLM (Qwen2.5-Omni) for Track 3 similarity — one
# SEPARATE adapter per metric, initialised from the released SQ-LLM checkpoint.
# Speech encoder frozen (--freeze_vit), direct "Score: X" supervision, no CoT,
# no GRPO. Mirrors SpeechLLM-as-Judges/script/qwenomni_train_ratio.sh but for a
# single GPU and our data.
#
# Usage:
#   bash pipeline-finetune.sh              # both metrics, sequentially
#   bash pipeline-finetune.sh spk_sim      # just one
# set -euo pipefail
module load miniconda/3 && conda activate speecheval
# swift sft pulls in deepspeed via accelerate; its import needs CUDA_HOME set,
# so load a CUDA toolkit (inference/zero-shot did not need this).
module load cudatoolkit/12.6

export DATA_ROOT=../data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=../data/train.csv
HERE=$(pwd)
SLM=$HERE/SpeechLLM-as-Judges
INIT_CKPT=$SLM/checkpoint            # LoRA is added on top of this merged model
SFTDIR=$HERE/sft                     # generated train/val jsonl
MODELDIR=$HERE/models                # trained adapters land here
mkdir -p "$SFTDIR" "$MODELDIR"

EPOCHS=${EPOCHS:-3}                  # 2,531 unique pairs -> keep it modest
EVAL_SAVE_STEPS=${EVAL_SAVE_STEPS:-200}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
METRICS=${1:-"spk_sim acc_sim"}

for M in $METRICS; do
  log "==================== FINE-TUNE: $M ===================="

  # 1. Build train/val jsonl (pair-level split, listener-wise rows)
  log "[$M] building SFT data ..."
  python csv_to_sft.py --mode train --data-root "$DATA_ROOT" --target-metric "$M" \
      --csv "$TRAIN_CSV" \
      --out-train "$SFTDIR/$M.train.jsonl" \
      --out-val   "$SFTDIR/$M.val.jsonl"

  OUT="$MODELDIR/sft_$M"
  log "[$M] launching swift sft -> $OUT ..."

  # 2. LoRA SFT, single GPU, encoder frozen, init from SQ-LLM checkpoint.
  #    Inherit CUDA_VISIBLE_DEVICES from Slurm (falls back to 0 interactively).
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  NPROC_PER_NODE=1 \
  VIDEO_MAX_PIXELS=50178 \
  FPS_MAX_FRAMES=12 \
  MAX_PIXELS=1003520 \
  ENABLE_AUDIO_OUTPUT=0 \
  swift sft \
      --model "$INIT_CKPT" \
      --dataset "$SFTDIR/$M.train.jsonl" \
      --val_dataset "$SFTDIR/$M.val.jsonl" \
      --tuner_type lora \
      --torch_dtype bfloat16 \
      --num_train_epochs "$EPOCHS" \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 16 \
      --learning_rate 1e-4 \
      --lora_rank 8 \
      --lora_alpha 32 \
      --target_modules all-linear \
      --freeze_vit true \
      --gradient_checkpointing true \
      --eval_steps "$EVAL_SAVE_STEPS" \
      --save_steps "$EVAL_SAVE_STEPS" \
      --save_total_limit 3 \
      --logging_steps 10 \
      --max_length 2048 \
      --output_dir "$OUT" \
      --warmup_ratio 0.05 \
      --dataloader_num_workers 4 \
      --seed 42 \
      --dataset_shuffle true

  log "[$M] done. Best adapter under $OUT (see the printed 'best_model_checkpoint')."
done

log "ALL DONE. Next: run inference with the trained adapter — see README (fine-tuned inference)."
