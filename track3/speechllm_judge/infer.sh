# Local swift-infer wrapper (keeps upstream SpeechLLM-as-Judges/script pristine).
# Same knobs as SpeechLLM-as-Judges/script/inference.sh, plus:
#   --repetition_penalty  -> stops the greedy (temperature 0) repetition loops
#                            that ran off to the 512-token cap without a score.
# Override defaults via env, e.g.  MAX_NEW_TOKENS=640 REP_PENALTY=1.15 bash infer.sh ...
#
# usage: bash infer.sh <checkpoint> <val_dataset.jsonl> <result_path.jsonl>
final_ckpt=$1
val_data_path=$2
output_jsonl=$3

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
REP_PENALTY=${REP_PENALTY:-1.1}

VIDEO_MAX_PIXELS=50178 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
swift infer \
    --model "$final_ckpt" \
    --val_dataset "$val_data_path" \
    --temperature 0 \
    --repetition_penalty "$REP_PENALTY" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --write_batch_size 50 \
    --max_batch_size 8 \
    --result_path "$output_jsonl"
