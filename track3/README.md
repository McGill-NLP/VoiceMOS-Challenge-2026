### 260728-v1 using v1 code but with "--lambda-moe-aux", type=float, default=0.05,
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim \
    --seed 47 --freeze-steps 5000 --wean-max-alpha 1.0 \
    --outdir /home/mila/j/jeony/scratch/voicemos_v2/track3/260728_code/egs/submissions/260728-v1-dual_encoder_wean_spk_sim

uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint home/mila/j/jeony/scratch/voicemos_v2/track3/260728_code/egs/submissions/260728-v1-dual_encoder_wean_acc_sim/model_v2_final.pt \
    --target-metric acc_sim \
    --out home/mila/j/jeony/scratch/voicemos_v2/track3/260728_code/egs/submissions/260728-v1-dual_encoder_wean_acc_sim/answer.txt


{
    "TRACK3_SPK_UTT_SRCC": 0.4812781689830857,
    "TRACK3_ACC_UTT_SRCC": 0.5021186275218261
}
