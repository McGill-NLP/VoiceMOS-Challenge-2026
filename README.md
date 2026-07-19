# VoiceMOS Challenge 2026

## Baseline results

### Speaker similarity, dev set

|            | UTT-MSE | UTT-LCC | UTT-SRCC | SYS-MSE | SYS-LCC | SYS-SRCC |
|------------|---------|---------|----------|---------|---------|----------|
| Baseline 1 |  12.032 |   0.529 |    0.432 |  11.590 |   0.848 |    0.809 |
| Baseline 2 |   0.438 |   0.511 |    0.451 |   0.069 |   0.916 |    0.860 |

### Accent similarity, dev set

|            | UTT-MSE | UTT-LCC | UTT-SRCC | SYS-MSE | SYS-LCC | SYS-SRCC |
|------------|---------|---------|----------|---------|---------|----------|
| Baseline 1 |  11.997 |   0.448 |    0.369 |  11.606 |   0.809 |    0.749 |
| Baseline 2 |   0.418 |   0.465 |    0.440 |   0.060 |   0.902 |    0.861 |



### 260719 - v1.1: freeze 3000 steps then unfreeze

```uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim \
    --outdir egs/spk_sim_v2 --freeze-steps 3000 --backbone-lr-mult 0.1
```

```uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric acc_sim \
    --outdir egs/spk_sim_v2 --freeze-steps 3000 --backbone-lr-mult 0.1
```

```uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/spk_sim_freeze3000/model_v2_final.pt \
    --target-metric spk_sim \
    --out egs/spk_sim_freeze3000/spk_v2_freeze_dev.csv
```

```uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/acc_sim_freeze3000/model_v2_final.pt \
    --target-metric acc_sim \
    --out egs/acc_sim_freeze3000/acc_v2_freeze_dev.csv
```

``` scores
{
    "TRACK3_SPK_UTT_SRCC": 0.45535613582560736,
    "TRACK3_ACC_UTT_SRCC": 0.4219199387832151
}
```




### 260719-v1.2-jointlyTrained: freeze 3000 steps then unfreeze

```
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric both \
    --outdir egs/spk_sim_v2 --freeze-steps 3000 --backbone-lr-mult 0.1
```
```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v1.2-jointlyTrained/spk_sim_v2/model_v2_final.pt \
    --target-metric both \
    --out /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v1.2-jointlyTrained/answer.txt
```

``` 
{
    "TRACK3_SPK_UTT_SRCC": 0.4615584754286811,
    "TRACK3_ACC_UTT_SRCC": 0.4347331322389782
}
```



### 260719-v2.1: freeze 5000 steps then unfreeze
```uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim \
    --outdir egs/spk_sim_freeze5000 --freeze-steps 5000 --backbone-lr-mult 0.1
```

```uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric acc_sim \
    --outdir egs/acc_sim_freeze5000 --freeze-steps 5000 --backbone-lr-mult 0.1
```

```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/spk_sim_freeze5000/model_v2_final.pt \
    --target-metric spk_sim \
    --out egs/spk_sim_freeze5000/spk_v2_freeze5000_dev.csv
```

```uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/acc_sim_freeze5000/model_v2_final.pt \
    --target-metric acc_sim \
    --out egs/acc_sim_freeze5000/acc_v2_freeze5000_dev.csv
```

```
{
    "TRACK3_SPK_UTT_SRCC": 0.47237322880386984,
    "TRACK3_ACC_UTT_SRCC": 0.4499761799187876
}
```


### 260719-v2.2-jointlyTrained: freeze 5000 steps then unfreeze

```
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric both \
    --outdir egs/submissions/260719-v2.2-jointfreeze5000 --freeze-steps 5000 --backbone-lr-mult 0.1
```
```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v2.2-jointfreeze5000/model_v2_final.pt \
    --target-metric both \
    --out /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v2.2-jointfreeze5000/answer.txt
```

```score:
{
    "TRACK3_SPK_UTT_SRCC": 0.44134390337419405,
    "TRACK3_ACC_UTT_SRCC": 0.41606139036368656
}
```

### 260719-v3.1: freeze 10000 steps then unfreeze
```uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim \
    --outdir egs/spk_sim_freeze10000 --freeze-steps 10000 --backbone-lr-mult 0.1
```

```uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric acc_sim \
    --outdir egs/acc_sim_freeze10000 --freeze-steps 10000 --backbone-lr-mult 0.1
```

```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/spk_sim_freeze10000/model_v2_final.pt \
    --target-metric spk_sim \
    --out egs/spk_sim_freeze10000/spk_v2_freeze10000_dev.csv
```

```uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/acc_sim_freeze10000/model_v2_final.pt \
    --target-metric acc_sim \
    --out egs/acc_sim_freeze10000/acc_v2_freeze10000_dev.csv
```


```scores:
{
    "TRACK3_SPK_UTT_SRCC": 0.45302064288729904,
    "TRACK3_ACC_UTT_SRCC": 0.4697257775601953
}
```


### 260719-v3.2-jointlyTrained: freeze 10000 steps then unfreeze

```
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric both \
    --outdir egs/submissions/260719-v2.2-jointfreeze10000 --freeze-steps 10000 --backbone-lr-mult 0.1
```
```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v2.2-jointfreeze10000/model_v2_final.pt \
    --target-metric both \
    --out /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v2.2-jointfreeze10000/answer.txt
```


#### Notes:
```
zip -j track3-260719-v1.2.zip /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v1.2-jointlyTrained/answer.txt
```
