# swift-infer wrapper for a FINE-TUNED LoRA adapter (base model + --adapters).
# The adapter records its base (the SQ-LLM checkpoint) in adapter_config.json,
# but we pass --model explicitly too for robustness. Output is short and
# deterministic ("Score: X"), so no repetition_penalty and a small token budget.
#
# usage: bash infer_ft.sh <base_ckpt> <adapter_dir> <val.jsonl> <result.jsonl>
base_ckpt=$1
adapter=$2
val_data_path=$3
output_jsonl=$4

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}

VIDEO_MAX_PIXELS=50178 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
swift infer \
    --model "$base_ckpt" \
    --adapters "$adapter" \
    --val_dataset "$val_data_path" \
    --temperature 0 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --write_batch_size 50 \
    --max_batch_size 8 \
    --result_path "$output_jsonl"
