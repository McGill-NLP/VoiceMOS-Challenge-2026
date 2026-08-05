# Track 3 — unified model

One training script that combines the ingredients that helped individually on the other
branches, so they can be stacked and ablated from the command line. See
[../../BRANCHES.md](../../BRANCHES.md) for what each ingredient did on its own.

| Axis | Flag | Taken from |
|---|---|---|
| Backbone | `--encoder`, composable with `+` | `dev.dg/contrastive` (`../encoders/`) |
| Prediction head | `--head mlp\|moe` | `dev.yj/empirical`, `dev.yj-v2` |
| Freeze schedule | `--freeze-steps`, `--backbone-lr-mult` | `dev.yj/empirical`, `dev.yj-v2` |
| Objective | `--objective mse\|corn\|coral` | `dev.ap/CORN` (`../corn-and-coral/`) |
| Contrastive auxiliary | `--lambda-rnc` | `dev.dg/contrastive` (`../rank-n-contrast/`) |

Defaults reproduce the official Baseline 2 recipe with an encoder that actually trains:

```bash
python finetune.py --data-root $DR --target-metric spk_sim --outdir egs/base
# = --encoder ecapa-voxceleb --head mlp --objective mse --lambda-rnc 0
#   batch 16, AdamW 1e-3 over one parameter group, 20,000 steps, MSE on per-pair means
```

Verified: with `--head mlp --objective mse` the head plus projection is **0.1151M
parameters, exactly the official baseline's**, so `base` is a like-for-like reference and
not an approximation of one.

## Architecture

Everything up to the interaction vector is the baseline's, unchanged:

```
wav_a → backbone → Linear(d, 256) → L2norm ─┐
                                            ├→ [e_a, e_b, |e_a−e_b|, e_a⊙e_b]  (1024-d)
wav_b → backbone → Linear(d, 256) → L2norm ─┘         │
                                                      ├──────────────→ RnC auxiliary
                                                      ↓
                                         trunk (MLP | MoE) → task layer → raw
                                                                          ↓
                                                             decode_score → 1..5
```

The head is split into a **trunk** (the part that varies with `--head`) and a **task
layer** (the part that varies with `--objective`), which is what makes the two axes
independent:

| `--objective` | trunk output | task layer | raw shape |
|---|---|---|---|
| `mse` | 1 | `Tanh·2+3` range clipping | `[B]` |
| `corn` | `--ordinal-dim` (128) | `Linear(128, K−1)` | `[B, 4]` conditional logits |
| `coral` | `--ordinal-dim` (128) | `CoralLayer(128, K)` | `[B, 4]` marginal logits |

`decode_score` maps any of them to a continuous 1–5 rating as the expected value
`E[y] = 1 + Σ_k P(y > k)`. We never threshold to a discrete class — the targets are *mean*
listener ratings, so 3.6 is a meaningful value and rounding it away discards signal.

Two details worth knowing:

- **The CORAL layer sits after the mixture, not inside it.** CORAL's rank consistency is a
  property of its architecture (one shared weight vector, K−1 ordered biases). Mixing
  per-expert CORAL logits would not preserve it, so the MoE produces a feature vector and
  a single CORAL layer reads it.
- **Soft ordinal targets are the default.** Per-pair means are fractional, and CORN/CORAL
  need integer classes. `soft_survival_targets` encodes a rating as a two-point mixture
  between `floor(t)` and `ceil(t)`; `--hard-labels` rounds instead. Verified: on integer
  targets the soft losses reproduce `coral_pytorch`'s `corn_loss` and `coral_loss` to
  `0.00e+00`.

## Freezing

`--freeze-steps N` keeps the backbone frozen for the first N optimizer steps and then
unfreezes it once, so a randomly initialised head settles before its gradients reach a
pretrained backbone. `--freeze-encoder` never unfreezes (the frozen control).

Freezing sets `requires_grad=False` **and** pins the backbone to `eval()`. Both are
needed: ECAPA has 31 `BatchNorm1d` layers whose running statistics keep drifting on every
forward pass in train mode even when the weights cannot move, so a backbone frozen by
`requires_grad` alone still changes its own embeddings. That is the defect that made the
official baseline's frozen arm irreproducible ([BRANCHES.md §3](../../BRANCHES.md)).
`model.train()` re-asserts the freeze, so it cannot be undone by accident.

The backbone stays in a parameter group throughout; it simply receives no gradient until
the unfreeze step, so the optimizer never needs rebuilding.

Learning rates: `--encoder-lr` sets the backbone rate explicitly, `--backbone-lr-mult`
sets it as a multiple of `--lr` (dev.yj used `0.1`), and if neither is given there is a
**single parameter group at `--lr`** — the baseline recipe, which is also the strongest
configuration measured so far.

## Rank-N-Contrast

`--lambda-rnc F` adds `F · rnc_loss(interaction, targets)` to whichever primary objective
is active, on the same interaction vector the head consumes — the RNC paper prescribes no
separate projection head for the contrastive term.

**This is not the paper's protocol.** RNC is two-stage: learn the representation, freeze
it, then fit a head. Here it is a joint auxiliary, which is what makes it composable with
`--objective` and `--head`. If you want the original protocol, `../rank-n-contrast/` still
implements it.

**Batch size matters more than the weight.** RNC ranks every sample against every other
sample *in its batch*, so gradient accumulation does not help it and batch 16 gives it
very little to rank; `../rank-n-contrast/` used 96. The script warns below 32. ERes2NetV2
will OOM long before that — it OOMs at plain batch 16 in full fine-tuning — so use a small
`--batch-size` there or skip the RNC arms for that encoder.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--encoder` | `ecapa-voxceleb` | `python encoders.py --list`; combine with `+` |
| `--embedding-dim` | 256 | projection width after the backbone |
| `--head` | `mlp` | `mlp` or `moe` |
| `--hidden-dim` | 64 | hidden width inside the trunk |
| `--num-experts` | 2 | MoE only; dev.yj found 3 experts worse than 2 |
| `--top-k` | unset | unset = dense softmax gating (dev.yj's choice, avoids expert collapse) |
| `--lambda-moe-aux` | 0.01 | load-balancing auxiliary weight |
| `--ordinal-dim` | 128 | trunk width feeding CORN/CORAL |
| `--objective` | `mse` | `mse`, `corn`, `coral` |
| `--hard-labels` | off | round per-pair means to integer classes |
| `--lambda-rnc` | 0.0 | 0 disables the auxiliary entirely |
| `--rnc-temperature` | 2.0 | paper default |
| `--rnc-label-diff` / `--rnc-feature-sim` | `l1` / `l2` | paper defaults |
| `--freeze-steps` | 0 | 0 trains the backbone from step 1 |
| `--freeze-encoder` | off | frozen control |
| `--encoder-lr` / `--backbone-lr-mult` | unset | unset = one group at `--lr` |
| `--batch-size` / `--accumulate-steps` | 16 / 1 | |
| `--train-steps` / `--save-steps` | 20000 / 5000 | |
| `--lr` / `--weight-decay` / `--grad-clip` | 1e-3 / 0.0 / 1.0 | |
| `--dev-csv` / `--eval-steps` / `--best-metric` | none / 500 / `srcc_sys` | dev scored during training |
| `--seed` | 1337 | |

## What each run records

Every run writes four things next to its checkpoints:

- `run_config.json` — every flag, for reproducing or grouping the run.
- `dev_log_<metric>.csv` — one row per evaluation: step, per-term training losses
  (`train_task`, `train_rnc`, `train_moe_aux`) and all six dev metrics. An arm that looks
  flat can then be diagnosed as "the auxiliary is doing nothing" rather than guessed at.
- `tensorboard/` — TensorBoard events (below).
- The **printed metric block**, in `calculate_metrics.py`'s exact layout, at every
  evaluation. That is what makes the Slurm log readable on its own:

```
==============================================
Results for SPK_SIM  [dev @ step 6]
==============================================
Evaluated Pairs   : 600
Evaluated Systems : 23
----------------------------------------------
[UTTERANCE LEVEL]
MSE  : 0.3322
LCC  : 0.5527
SRCC : 0.5901
----------------------------------------------
[SYSTEM LEVEL]
MSE  : 0.2419
LCC  : 0.3819
SRCC : 0.5630
==============================================
```

Matching the standalone tool's layout means the in-training number and the post-hoc
`calculate_metrics.py` number can be compared without re-reading either. The step-0 block
is tagged `(untrained reference)`.

### TensorBoard

```bash
tensorboard --logdir track3/unified/egs      # every arm shows up as its own run
```

Scalars written, all from the same `compute_metrics` the standalone tool uses:

| Tag | When |
|---|---|
| `dev_utt/{mse,lcc,srcc}`, `dev_sys/{mse,lcc,srcc}` | every `--eval-steps` |
| `dev_best/<best-metric>` | every evaluation, the running best |
| `train/task_loss`, `train/total_loss` | every `--log-every` steps (default 50) |
| `train/rnc_loss` | only when `--lambda-rnc > 0` |
| `train/moe_aux_loss` | only when `--head moe` |
| `train/grad_norm`, `train/lr_group{0,1}` | every `--log-every` steps |
| `train/backbone_trainable` | 0/1 trace of the two-phase schedule |

`train/backbone_trainable` is the one to check first when reading a `--freeze-steps` run:
it puts the exact unfreeze step on the same time axis as the loss, so a jump can be
attributed rather than guessed at. `lr_group0` is the backbone and `lr_group1` the head
whenever `--encoder-lr` or `--backbone-lr-mult` is set; with neither there is a single
group.

The full `run_config` also lands in TensorBoard's TEXT tab. Logging is optional and never
fatal — `--no-tensorboard` disables it, `--tensorboard-dir` relocates it, and a missing
`tensorboard` install or an unwritable directory degrades to a warning rather than killing
a run that costs hours of GPU.

## Usage

```bash
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
LABELS=../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv

# reference arm
python finetune.py --data-root $DR --target-metric spk_sim --outdir egs/base \
    --dev-csv $LABELS --dev-data-root $DR

# everything stacked
python finetune.py --data-root $DR --target-metric spk_sim --outdir egs/stack \
    --encoder eres2netv2 --head moe --objective corn \
    --freeze-steps 2000 --backbone-lr-mult 0.1 \
    --lambda-rnc 0.5 --batch-size 8 --accumulate-steps 2 \
    --train-steps 8000 --dev-csv $LABELS --dev-data-root $DR

# predict; architecture is read back from the checkpoint
python inference.py --data-root $DR --csv-path $DR/sets/dev.csv \
    --checkpoint egs/stack/model_best_spk_sim.pt --out egs/stack/dev_spk_sim.csv
python calculate_metrics.py --prediction-csv egs/stack/dev_spk_sim.csv \
    --ground-truth-csv $LABELS
```

Note the data root: use the **training** distribution for train and dev alike. The
evaluation-phase distribution is missing all 600 `sys019` reference wavs, so inference
against it silently drops every row.

## Sweeping

[`../jobs/voicemos-track3-unified-sweep.sh`](../jobs/voicemos-track3-unified-sweep.sh)
runs a **ladder**, not a grid: a reference arm, then one ingredient at a time, then the
stack. That is what makes a gain attributable — the full grid is 288 runs.

```bash
sbatch track3/jobs/voicemos-track3-unified-sweep.sh                     # ecapa, both targets
for E in ecapa-voxceleb eres2netv2 commonaccent-ecapa; do               # sweep encoders
    sbatch --export=ALL,ENCODER=$E track3/jobs/voicemos-track3-unified-sweep.sh
done
```

Arms: `base`, `moe`, `freeze`, `corn`, `coral`, `rnc`, `moe-freeze`, `stack`,
`stack-rnc`. Default set is `base moe freeze corn stack stack-rnc` — `corn` is in it
because `stack` contains it, and without the single-ingredient arm a `stack` gain cannot
be attributed to the head, the schedule or the objective.

**One job runs `len(CONFIGS) × len(METRICS)` experiments** — 6 × 2 = 12 by default, 18 if
you pass all nine arms. Measured per-arm cost on an L40S with ECAPA:

| Arm | s/step | Steps | Peak GiB | Wall |
|---|---|---|---|---|
| `base` / `moe` / `corn` / `stack` | 0.192 | 8,000 | 9.3 | 29 min |
| `freeze` | 0.161 | 8,000 | 2.6 frozen → 9.3 | 25 min |
| `rnc` / `stack-rnc` (batch 32) | 0.375 | 4,000 | 21.2 | 28 min |

So the default 12 runs are ~5.6 h for ECAPA, inside the 12 h wall. **ERes2NetV2 is ~3×
slower per optimizer step and the same ladder will not fit** — split it by target, e.g.
`--export=ALL,ENCODER=eres2netv2,METRICS=spk_sim`.

Two deliberate choices in there:

- **`RNC_BATCH` is 32, not larger.** At batch 64 the arm peaks at 42.2 GiB of a 46 GiB
  card, and since repetitive padding stretches every clip in a batch up to the longest
  one, that OOMs when a long clip turns up rather than failing at step 1.
- **The RnC arms run half the steps.** They need the larger batch for the loss to have
  anything to rank, which at equal step counts would hand them twice the data exposure of
  every other arm (8,000 × 32 = 256k sample presentations vs 128k). A win would then be
  unattributable — more RnC, or just more epochs? They are matched on samples seen
  instead. `RNC_STEPS=$TRAIN_STEPS` compares at equal steps.

The job prints all six metrics per arm at the end, because the challenge's final ranking
combines several of them and reading a single column will mislead you — on the existing
runs the UTT-SRCC and SYS-SRCC orderings disagree sharply.

## Layout

```
encoders.py            Encoder registry, unchanged copy from ../encoders/.
eres2netv2/            ERes2NetV2 implementation, unchanged copy from ../encoders/.
heads.py               MLPTrunk, MoETrunk, Head (trunk + task layer).
objectives.py          task_loss (mse/corn/coral, soft + hard), decode_score, rnc_loss.
model.py               UnifiedModel: encoder + interaction + head, freeze control.
finetune.py            The training script. All axes are flags here.
inference.py           Rebuilds the architecture from the checkpoint's config.
calculate_metrics.py   Unchanged copy from ../baseline/.
make_eval_gt.py        Averages a listener-wise split into one row per pair.
```

## Deliberately not included

- **Listener modelling** — a listener embedding and bias head, as in `dev.yj/empirical`
  and UTMOS. It is the single largest ablation effect in the UTMOS paper and Track 3 gives
  25 listeners with ~550 ratings each. It needs the listener-wise rows rather than per-pair
  means, so it changes the data pipeline rather than composing as a flag. The most valuable
  thing to add next.
- **The UTMOS objective** (contrastive + clipped MSE). It would slot into
  `objectives.task_loss` as two more choices; `../utmos-approach/` has the verified port.
  Its contrastive-alone arm was the best objective on that branch.
- **Post-hoc recalibration.** `../utmos-approach/recalibrate.py` fits an affine rescale on
  train and applies it to a prediction CSV. It leaves LCC and SRCC untouched by
  construction and only repairs MSE, so it is worth running over any arm here whose
  correlations are good but whose MSE is not.
- **Joint `--target-metric both`**, as in `dev.yj/empirical`. Its results there were mixed:
  joint beat separate on `spk_sim` and lost on `acc_sim`.
