# Track 3 — baseline recipe with pluggable encoders

Same approach as [../baseline/](../baseline/) — fine-tune a pretrained utterance encoder plus a
projection head on the Track 3 pairs — but the backbone is selectable instead of hard-coded to
SpeechBrain's ECAPA. Everything downstream of the encoder (projection, 4-way interaction vector,
range clipping, repetitive padding, per-pair score averaging, AdamW + MSE, fixed step count) is
unchanged, so encoder swaps are the only variable when comparing against the baseline numbers.

## Available encoders

```bash
python encoders.py --list
```

| Name | Params | Source | Notes |
|---|---|---|---|
| `ecapa-voxceleb` | 22.1M | `speechbrain/spkrec-ecapa-voxceleb` | The baseline's encoder. Speaker verification on VoxCeleb. |
| `commonaccent-ecapa` | 20.8M | `Jzuluaga/accent-id-commonaccent_ecapa` | ECAPA fine-tuned on CommonVoice for 16-way English accent ID. |
| `eres2netv2` | 17.9M | `iic/speech_eres2netv2_sv_zh-cn_16k-common` | ERes2NetV2, 200k speakers. EER 0.61% VoxCeleb1-O. |
| `eres2netv2-w24s4ep4` | ~24M | `iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common` | Wider ERes2NetV2 variant. |

Encoders compose with `+`. Each branch is L2-normalised before concatenation:

```bash
--encoder eres2netv2+commonaccent-ecapa    # 384-d, a speaker branch and an accent branch
```

Weights are downloaded on first use to `~/.cache/vmc2026-track3-encoders`
(override with `--cache-dir` or `$VMC_ENCODER_CACHE`). ERes2NetV2 comes from ModelScope with a
Hugging Face mirror as fallback; no `modelscope` package is required.

### What the two new encoders are

**ERes2NetV2** (Chen et al., Interspeech 2024 — paper in [../papers/](../papers/)) improves ERes2Net
for *short* utterances via two changes. Bottom-up dual-stage feature fusion (BDFF) keeps only the
stage-3 ↔ stage-4 AFF fusion and drops the stage-1/2/3 global fusion as redundant. Bottleneck-like
local feature fusion (BLFF) does expand-reduce-expand inside each block, widening channels while
cutting parameters. Result: 17.8M params, EER 0.61% / 0.98% / 1.48% on full / 3s / 2s VoxCeleb1-O,
against ECAPA's 1.27% at 3s. Given the Track 3 clips are a few seconds long, the short-duration
gain is the reason to try it.

It takes 80-dim Kaldi FBank with per-utterance mean normalisation, not raw waveform. That happens
inside `ERes2NetV2Encoder`, so the interface stays waveform-in / embedding-out like every other
encoder. `torchaudio.compliance.kaldi.fbank` only accepts one utterance at a time, so features are
computed per row on the *unpadded* waveform and repetitively padded in the feature domain — this
keeps CMN statistics free of padding and avoids the mid-frame discontinuity that waveform-domain
repetitive padding would introduce.

The checkpoint is trained on a 200k-speaker Chinese-English corpus. Speaker embeddings transfer
across languages reasonably well, but this is English data, so treat it as an empirical question.

**CommonAccent ECAPA** (Zuluaga-Gomez et al., Interspeech 2023) is the same ECAPA architecture as
the baseline's, fine-tuned on CommonVoice to classify 16 English accents (england, us, canada,
australia, indian, scotland, ireland, african, malaysia, newzealand, southatlandtic, bermuda,
philippines, hongkong, wales, singapore). Its hyperparams layout is identical to the baseline
encoder's, so it needs no special handling.

The motivation is in [../papers/IDEAS.md](../papers/IDEAS.md): a VoxCeleb speaker-ID encoder is
trained to be *invariant* to accent, which is why the zero-shot baseline emits byte-identical
predictions for `spk_sim` and `acc_sim`. This encoder is discriminative in exactly the dimension
the baseline discards. `SpeechBrainEncoder.classify()` also exposes the 16-way accent posteriors,
which are worth trying as an explicit feature (e.g. JS divergence between the two utterances)
independently of fine-tuning.

## Usage

Identical to the baseline, plus `--encoder`:

```bash
# zero-shot cosine similarity
python inference.py --data-root $DATA_ROOT --csv-path $DATA_ROOT/sets/dev.csv \
    --encoder eres2netv2 --out egs/zero_shot_eres2netv2/spk_dev.csv

# fine-tune
python finetune.py --data-root $DATA_ROOT --target-metric spk_sim \
    --encoder eres2netv2 --outdir egs/eres2netv2_spk

# inference — encoder and target metric are read back from the checkpoint
python inference.py --data-root $DATA_ROOT --csv-path $DATA_ROOT/sets/dev.csv \
    --checkpoint egs/eres2netv2_spk/finetuned_model_spk_sim_final.pt \
    --out egs/eres2netv2_spk/spk_dev.csv

python calculate_metrics.py --prediction-csv  egs/eres2netv2_spk/spk_dev.csv \
                            --ground-truth-csv $DATA_ROOT/sets/dev_with_labels.csv
```

Checkpoints are saved as `{"config": ..., "state_dict": ...}`, so inference reconstructs the right
architecture and writes the right `pred_*` column without being told. Baseline-format checkpoints
(bare `state_dict`) still load: the backbone key prefix is remapped and the head name is recovered
from the keys.

### Extra flags beyond the baseline

| Flag | Purpose |
|---|---|
| `--encoder NAME` | Backbone to use. Combine with `+`. |
| `--freeze-encoder` | Train only the projection + MLP head (~0.12M params). |
| `--encoder-lr LR` | Separate, usually lower, LR for the backbone. Head keeps `--lr`. |
| `--encoder-checkpoint PATH` | Local backbone weights instead of downloading. |
| `--cache-dir DIR` | Where to cache downloaded weights. |
| `--train-csv PATH` | Train on a custom split (e.g. `../corn-and-coral/data/train.csv`). |
| `--num-workers`, `--seed` | Were hard-coded in the baseline. |

## Slurm

Two job scripts in [../jobs/](../jobs/), one per new encoder, each covering both targets:

```bash
sbatch track3/jobs/voicemos-track3-encoders-eres2netv2.sh          # ~6 h
sbatch track3/jobs/voicemos-track3-encoders-commonaccent-ecapa.sh  # ~2 h
```

Each trains `{spk_sim, acc_sim} × {frozen, encoder-lr 1e-5, encoder-lr 1e-4}` on the **complete
official `sets/train.csv`**, scoring the labelled dev set throughout training.

### Watching a run

The labelled dev set (`vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv`, 600
pairs, 23 systems) is scored every `--eval-steps` optimizer steps, so a run reports as it goes:

```
[dev @ step 1500] mse_utt=0.5198 lcc_utt=0.2693 srcc_utt=0.2721 mse_sys=0.2547 lcc_sys=0.6597 srcc_sys=0.5484
[dev @ step 1500] new best srcc_sys=0.5484, saved model_best.
```

Each run writes into its `egs/<tag>/` directory:

| File | Contents |
|---|---|
| `dev_log_<metric>.csv` | one row per evaluation: `step, train_mse`, then all six dev metrics |
| `model_best_<metric>.pt` | checkpoint with the best dev `--best-metric` (default `srcc_sys`) |
| `dev_<metric>.csv` | predictions from that checkpoint, submission format |

A step-0 evaluation runs before training as the reference point, but is excluded from
best-checkpoint tracking — a randomly initialised head can score well by chance.

The dev wavs resolve against the **training** distro: the eval distro ships without the 600
`sys019` reference wavs (they come with VCTK separately), while the train distro has every dev
wav. Hence `--dev-data-root $DR` in the job scripts.

**In-training scores are a monitoring signal, not the number of record.** Evaluation runs batched,
so the collater pads each clip to the batch maximum, while `inference.py` runs unpadded at batch
size 1. Utterance-level metrics track closely (measured `srcc_utt` 0.1691 vs 0.1696 on one
checkpoint), but system-level SRCC ranks only ~23 systems, so a hair of numerical difference can
swap two adjacent ones and shift it by ~0.01 (`srcc_sys` 0.4308 batched vs 0.4407 unbatched). The
scripts therefore re-score the selected checkpoint with `inference.py` + `calculate_metrics.py`
after training, and that is the number to quote. `--eval-batch-size 1` makes the two agree exactly,
at roughly 15× the evaluation cost.

### Standalone

```bash
python finetune.py --data-root $DR --train-csv $DR/sets/train.csv \
    --target-metric spk_sim --encoder eres2netv2 --encoder-lr 1e-5 --outdir egs/run \
    --dev-csv $EVAL_DR/sets/dev_with_labels.csv --dev-data-root $DR \
    --eval-steps 250 --best-metric srcc_sys
```

For a held-out read on unseen systems and listeners, train on the local 75% split and evaluate on
`dev-ID` / `dev-OOD` instead:

```bash
sbatch --export=ALL,USE_LOCAL_SPLITS=1 track3/jobs/voicemos-track3-encoders-eres2netv2.sh
```

`FROZEN_STEPS`, `FT_STEPS` and `SAVE_STEPS` are overridable the same way, which is the cheap way to
smoke-test a change (`FROZEN_STEPS=2 FT_STEPS=2`).

### Measured cost on an L40S (46 GB)

| Encoder | Mode | Batch | s/step | Peak GPU |
|---|---|---|---|---|
| `eres2netv2` | frozen | 16 | 0.245 | 3.85 GiB |
| `eres2netv2` | full fine-tune | 4 (×4 accum) | 0.147 | **13.97 GiB** |
| `eres2netv2` | full fine-tune | 8 | 0.293 | 30.70 GiB |
| `eres2netv2` | full fine-tune | 12 | 0.477 | 42.29 GiB |
| `eres2netv2` | full fine-tune | 16 | — | **OOM** |
| `commonaccent-ecapa` | frozen | 16 | 0.068 | 2.96 GiB |
| `commonaccent-ecapa` | full fine-tune | 16 | 0.157 | 11.18 GiB |

ERes2NetV2 **cannot** run the baseline's batch size of 16. It is a 2D CNN with stride 1 in stage 1,
so activations scale with batch × frames, and repetitive padding stretches every clip in a batch to
the longest one (clips run 2.5–9.0 s). The job script uses batch 4 with 4 accumulation steps:
identical throughput to batch 8 × 2 (0.588 s per optimizer step either way) but with 17 GiB more
headroom, and the effective batch stays at the baseline's 16.

## ⚠️ The baseline never actually fine-tuned its encoder

Worth knowing before you compare numbers. `../baseline/model.py` passes `freeze_ssl=False` with the
comment `# Fine-tuning everything`, but it never reaches the backbone:

```python
EncoderClassifier.from_hparams(source=model_name, run_opts={"device": "cpu"})
```

SpeechBrain's `Pretrained.__init__` defaults to `freeze_params=True`, which sets
`requires_grad=False` on every backbone parameter. Verified against the baseline directly:

```
total params      : 22.27M
trainable         :  0.12M
trainable in ECAPA:  0.00M   <-- freeze_ssl=False was passed
```

So published "Baseline 2" is a **frozen ECAPA with a trained 0.12M head**, not a fine-tuned
encoder. That is consistent with its results: utterance-level LCC barely moves from zero-shot
(0.529 → 0.511 for `spk_sim`), and the gains are almost entirely system-level — the head learned
to map a cosine-like feature into the 1–5 range.

This package passes `freeze_params=False` and makes freezing an explicit choice:

- **default** — the backbone really trains (22.27M / 20.89M / 17.97M trainable).
- **`--freeze-encoder`** — 0.12M trainable, i.e. what the baseline actually did.

Two consequences:

1. To reproduce the published baseline, use `--freeze-encoder`. Without it you are running a
   genuinely different (and much heavier) experiment.
2. The baseline's `--lr 1e-3` default was tuned for a 0.12M head. Applying it to a full pretrained
   backbone for 20k steps × bs16 (≈114 epochs over 2,800 pairs) will very likely destroy the
   pretrained representation. Start around `--encoder-lr 1e-5` with `--lr 1e-3`, per the schedule
   in Tseng et al. (1e-4 / 5e-5 / 1e-5).

## Verification

Checks run against the real distro (`vmc2026_track3_train_phase_distro_v3_syn`):

| Check | Result |
|---|---|
| `ecapa-voxceleb` zero-shot vs `../baseline/official-egs/zero_shot/spk_dev.csv` | max abs delta **3.9e-05** |
| Official fine-tuned checkpoint through this code vs its published csv | max abs delta **2.0e-04** |
| ERes2NetV2 checkpoint load | 0 missing / 0 unexpected keys, 17.86M params (paper: 17.8M) |
| Gradients reach the backbone (default, unfrozen) | ECAPA 138/139, CommonAccent 138/139, ERes2NetV2 290/290 tensors |
| Fine-tune → save → reload → infer → metrics | passes for all four encoder specs |

Deltas are float32 GPU nondeterminism, so the refactor is behaviour-preserving for the baseline path.

## Environment note

The audio in the distro is 24 kHz and is resampled to 16 kHz on load. In an env with
torchaudio ≥ 2.9, `torchaudio.load` dispatches to torchcodec, which may fail with
`libnppicc.so.12: cannot open shared object file`. Point the loader at the NVIDIA libs shipped
with torch:

```bash
export LD_LIBRARY_PATH=$(python -c "import site;print(site.getsitepackages()[0])")/nvidia/npp/lib:$LD_LIBRARY_PATH
```

This affects `../baseline/` identically; it is not specific to this package.

## Layout

```
encoders.py            encoder registry, wrappers, checkpoint fetching. `--list` / `--encoder X` to smoke-test.
model.py               Projection + SpeechEncoder + Model. Same as the baseline with a pluggable backbone.
finetune.py            Baseline training loop + encoder flags.
inference.py           Baseline inference + checkpoint-driven encoder/metric resolution.
calculate_metrics.py   Unchanged copy of ../baseline/calculate_metrics.py.
make_eval_gt.py        Averages a listener-wise split into one row per pair, for local evaluation.
eres2netv2/            Vendored from 3D-Speaker (Apache 2.0). Imports made relative; model code unmodified.
```

`eres2netv2/` is vendored rather than imported from `../papers/3D-Speaker` because `track3/papers`
is gitignored, and this directory should be self-contained.

## Adding another encoder

Implement `BaseEncoder` (set `output_dim`, implement `forward(waveform, lengths) -> (B, D)`) and add
an entry to `ENCODER_REGISTRY`. Nothing else changes. Candidates from
[../papers/IDEAS.md](../papers/IDEAS.md): ReDimNet, CAM++, WavLM-Large + x-vector head, XEUS.
