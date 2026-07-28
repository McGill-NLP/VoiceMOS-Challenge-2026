# The baseline system of the VoiceMOS Challenge 2026 Track 3

Contact for questions about this baseline: Wen-Chin Huang (Nagoya University) wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

This repository contains the baseline system for Track 3 of the VoiceMOS Challenge 2026.  
The task is to predict the speaker and accent similarity of a synthetic speech sample and a reference sample.

## Environment Setup

We use `uv` for environment management. We assume you are in a Linux environment. If you don't have `uv` installed, you can install it via `curl`.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

All commands below start with `uv run` and `uv` should take care of everything as it reads `uv.lock`. Please do note that my machine has CUDA version 12.6, so if your machine has a different version, you might need to delete `uv.lock`, modify `pyproject.toml` and let `uv run` take care the rest.

## Dataset instructions

Please download the dataset from CodaBench, decompress it and put it anywhere you like. We will assume the path to it is `DATA_ROOT` from now on.  

```bash
tree $DATA_ROOT

$DATA_ROOT
├── README
├── sets
│   ├── dev.csv
│   └── train.csv
└── wav
    ├── vmc2026-track3-sys001-utt003.wav
    ...
```

For the training phase, we provide training set waveform samples with the corresponding similarity scores, as well as development set waveform samples without the scores. Participants can submit their development set prediction results to the CodaBench system to obtain the performance.

In particular, `DATA_ROOT/sets/train.csv` has the following header:

`system_id,utterance_id,listener_id,wav_a_path,wav_b_path,spk_sim,acc_sim`

Please note that these are _listener-wise_ scores, so there will be multiple rows with the same sample pair (but with different `listener_id`, `spk_sim` and `acc_sim`).

`DATA_ROOT/sets/dev.csv` has the following header:

`system_id,utterance_id,wav_a_path,wav_b_path`

## Trained models

For participants' reference, trained models and their corresponding inference result csv files are in `official-egs`.

```bash
$ tree official-egs/

official-egs/
├── acc_sim_adamw_lr1e-3
│   ├── acc_step20000_dev.csv
│   ├── model_acc_sim_step20000.pt
├── spk_sim_adamw_lr1e-3
│   ├── model_spk_sim_step20000.pt
│   ├── spk_step20000_dev.csv
└── zero_shot
    ├── acc_dev.csv
    └── spk_dev.csv
```

## Baseline 1: zero-shot cosine similarity using pre-trained `speechbrain/spkrec-ecapa-voxceleb`

The first baseline is to calculate the cosine similarity of the embeddings of the two samples using the pre-trained `speechbrain/spkrec-ecapa-voxceleb`, which . No training here, so this is a zero-shot setting.

### Inference

Run the following command to conduct inference and obtain the resulting csv:

```bash
uv run python inference.py --data-root <DATA_ROOT> --csv-path <DATA_ROOT>/sets/dev.csv --out egs/zero_shot/spk_dev.csv
```
```bash (actual code to use)
uv run python inference.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_vctk --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv --out egs/zero_shot/spk_dev.csv
```


More details of `speechbrain/spkrec-ecapa-voxceleb` can be found here: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

## Baseline 2: fine-tune `speechbrain/spkrec-ecapa-voxceleb` with a projection head.

The second baseline is to fine-tune `speechbrain/spkrec-ecapa-voxceleb` with the provided training set. A projection head takes the embeddings of the two samples as input and outputs the similarity score. Techniques like range clipping as described in https://arxiv.org/abs/2104.03017 and repetitive padding as described in https://arxiv.org/abs/2103.00110 were used. Training was conducted with a batch size of 16, the AdamW optimizer with learning rate 0.001, and a fixed number of training steps of 20,000.

### Fine-tuning

Say we want to fine-tune a model to predict speaker similarity. Run the following command to perform fine-tuning:

```bash
uv run python finetune.py --data-root <DATA_ROOT> --target-metric spk_sim --outdir egs/spk_sim
```

```bash
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim --outdir egs/spk_simv2 --moe-spaced-init --verbose 2
```
uv run python finetune.py --data-root <DATA_ROOT> --target-metric spk_sim \
    --outdir egs/spk_sim_v2 --moe-spaced-init --verbose 2


Use `--outdir` to specify where the model checkpoints will be saved. To fine-tune a model to predict accent similarity, simply pass `--target-metric acc_sim`.

### Inference

Using the trained model (say put in `egs/spk_sim/model_spk_sim_step20000.pt`), run the following command to conduct inference and obtain the resulting csv:

```bash
uv run python inference.py --data-root <DATA_ROOT> --csv-path <DATA_ROOT>/sets/dev.csv --checkpoint egs/spk_sim/model_spk_sim_step20000.pt --out egs/spk_sim/spk_step20000_dev.csv
```

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

## Acknowledgement and citation

This repo was a subset of [sheet](https://github.com/unilight/sheet), an open-source repo for speech quality assessment research. In addition, Gemini 3.1 Pro and ChatGPT 5.5 were used to assist the implementation of this repo.



############## ORIGINAL:
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim --outdir egs/spk_sim

uv run python inference.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_vctk --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv --out egs/zero_shot/spk_dev.csv




#####################
Single task training:
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric acc_sim --outdir egs/acc_sim

uv run python inference.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev-OOD.csv \
    --checkpoint egs/spk_simv2/finetuned_model_spk_sim_final.pt --target-metric spk_sim --out egs/spk_simv2/dev-OOD.csv

python calculate_metrics.py --prediction-csv /home/mila/j/jeony/scratch/voicemos/track3/egs/spk_simv2/dev-OOD.csv --ground-truth-csv /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev-OOD.csv


### 260718
export HF_HOME="/home/mila/j/jeony/scratch/voicemos/huggingface"
export HF_HUB_CACHE="/home/mila/j/jeony/scratch/voicemos/huggingface/hub"

uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim \
    --outdir egs/spk_sim_v2 --lambda-rank 0.5


uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric acc_sim \
    --outdir egs/spk_sim_v2 --lambda-rank 0.5



uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev-ID.csv \
    --checkpoint egs/spk_sim_v2/model_v2_final.pt \
    --target-metric spk_sim \
    --out egs/spk_sim_v2/spk_v2_dev-ID.csv

python calculate_metrics.py --prediction-csv /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/spk_sim_v2/spk_v2_dev-ID.csv --ground-truth-csv /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev-ID.csv


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

'''scores:
{
    "TRACK3_SPK_UTT_SRCC": 0.45302064288729904,
    "TRACK3_ACC_UTT_SRCC": 0.4697257775601953
}
'''

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


```scores:
{
    "TRACK3_SPK_UTT_SRCC": 0.45985246526489576,
    "TRACK3_ACC_UTT_SRCC": 0.4090524132526068
}
```

### 260722-v1.1: separatelyTrained and freeze 5000 with MoE 2
```
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric both \
    --outdir egs/submissions260725-v1.1/joint
```


```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/submissions/260725-v1.1/joint/model_v2_final.pt \
    --target-metric both \
    --out egs/submissions/260725-v1.1/joint/answer.txt
```


#### scores for separately trained:
```
{
    "TRACK3_SPK_UTT_SRCC": 0.4762885787494583,
    "TRACK3_ACC_UTT_SRCC": 0.4752694342317759
}
```

#### scores for jointly trained:
```
{
    "TRACK3_SPK_UTT_SRCC": 0.483622001473416,
    "TRACK3_ACC_UTT_SRCC": 0.46046492002875905
}
```

### 260722-v1.2: separatelyTrained and freeze 5000 with MoE 3


``` 260722-v1.2-joint
too low
```

``` 260722-v1.2-separately
0.459 - too low
```

### 260722-v1.2: separatelyTrained and freeze 5000 with MoE 2 and Concordance Correlation Coefficient (CCC) loss
Note that when finetuning acc, architecture changes a bit, and scores increase - although it decreases for spk

```
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric spk_sim \
    --outdir egs/submissions/260722-v1.3-plusCCC/spk_sim
```

```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/submissions/260722-v1.3-plusCCC/spk_sim/model_v2_final.pt \
    --target-metric spk_sim \
    --out egs/submissions/260722-v1.3-plusCCC/spk_sim/answer.txt
```

```scores
{
    "TRACK3_SPK_UTT_SRCC": 0.4553844456761165,
    "TRACK3_ACC_UTT_SRCC": 0.4719290714947303
}
```



### 260723-v1.1: joint

```
uv run python finetune.py \
  --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric both \
  --outdir egs/submissions/260723-v1.1-joint_pertask \
  --freeze-steps-spk 5000 --freeze-steps-acc 10000 \
  --lambda-ccc-spk 0.0 --lambda-ccc-acc 1.0 \
  --num-experts 2 --moe-entropy-weight 0.01 \
  --eval-steps 500 --val-system-frac 0.1 --swa-frac 0.2 \
  --seed 42
```
```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/submissions/260723-v1.1-joint_pertask/model_v2_final.pt \
    --target-metric both \
    --out egs/submissions/260723-v1.1-joint_pertask/both/answer.txt
```


### 260725-v1.1: recheck-best - joint works the best (but no seed)
```
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric both \
    --outdir egs/submissions/260725-v1.1/joint
```


```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/submissions/260725-v1.1/joint/model_v2_final.pt \
    --target-metric both \
    --out egs/submissions/260725-v1.1/joint/answer.txt
```

``` scores:
{
    "TRACK3_SPK_UTT_SRCC": 0.4943996591824203,
    "TRACK3_ACC_UTT_SRCC": 0.47222416048439286
}
```



### 260725-v1.1: recheck-best - joint works the best (but with seed 42,47,52 중 47 works best)
moe 2 freeze 5000 joint

``` scores:
{
    "TRACK3_SPK_UTT_SRCC": 0.47785364596435687,
    "TRACK3_ACC_UTT_SRCC": 0.48573753484719634
}
```

### 260725-v1.2: seed 47 lambda_rank 0.3 moe 2 freeze until 5000 (separately trained)
lambda_rank 0.3 worse than not using lambda-rank. 

```scores
{
    "TRACK3_SPK_UTT_SRCC": 0.49169421367548805,
    "TRACK3_ACC_UTT_SRCC": 0.439819568323835
}
```

### 260725-v1.3: seed 47 moe 2 freeze until 5000 (separately trained)
1. spk uses lambda rank 0.3
2. acc does not

{
    "TRACK3_SPK_UTT_SRCC": 0.49169421367548805,
    "TRACK3_ACC_UTT_SRCC": 0.4520332123540985
}

### 260725-v1.4: seed 47 moe 2 freeze until 5000 (separately trained)
1. spk uses lambda rank 0.3 step 30k
2. acc does not


```
uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260725-v1.2/spkSeed47-lambaRank0.4-step30k/model_v2_final.pt \
    --target-metric spk_sim \
    --out /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260725-v1.2/spkSeed47-lambaRank0.4-step30k/answer.txt
```


### more changes:
uv run python finetune.py --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn --target-metric both \
    --seed 47 --freeze-steps 5000 --lambda-rank-spk 0.3 --lambda-rank-acc 0.0 \
    --outdir egs/submissions/260726-v1/joint_asymmetric_rank

uv run python inference.py \
    --data-root /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn \
    --csv-path /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv \
    --checkpoint egs/submissions/260726-v1/joint_asymmetric_rank/model_v2_final.pt \
    --target-metric both \
    --out egs/submissions/260726-v1/joint_asymmetric_rank/answer.txt


#### Notes:
```
zip -j track3-260719-v3.2.zip /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260719-v3.2-jointfreeze10000/answer.txt
```

python calculate_metrics.py --prediction-csv /home/mila/j/jeony/scratch/voicemos_v2/track3/egs/submissions/260725-v1.1-doNOTerase/joint-bestSoFar/answer.csv --ground-truth-csv /home/mila/j/jeony/scratch/voicemos/data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv
