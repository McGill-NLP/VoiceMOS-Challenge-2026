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
| Interaction vector | `--interaction` | this branch, see below |

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

## Encoders

`python encoders.py --list`. Two families behind one contract — `encoder(waveform, lengths)`
returns `(B, output_dim)`, and `--encoder` accepts any name below, or several joined by `+`.

| name | dim | params | pretraining |
|---|---|---|---|
| `ecapa-voxceleb` | 192 | 20.8M | speaker ID, VoxCeleb (the baseline's) |
| `commonaccent-ecapa` | 192 | 20.8M | accent ID, CommonVoice |
| `eres2netv2` | 192 | 17.8M | speaker ID, 3D-Speaker |
| `eres2netv2-w24s4ep4` | 192 | 53.5M | speaker ID, 3D-Speaker |
| `wavlm-base-plus-l{4,8,12}` | 768 | 37.7M at l4 | masked prediction + denoising, 94k h |
| `wavlm-large-l{4,8,24}` | 1024 | 63.5M at l4 | masked prediction + denoising, 94k h |
| `xlsr-300m-l{4,8,24}` | 1024 | 63.5M at l4 | wav2vec 2.0, 436k h, 128 languages |

The four speaker/accent-ID nets are trained to be *invariant* to the channel and phonetic
variation accent lives in; the SSL bundles retain it. That difference is why the SSL models
dominated the frozen-feature sweep in [../weak/](../weak/) — `WAVLM_LARGE_l4` + ridge scores
0.602 dev UTT-SRCC on `spk_sim`, above every fine-tuned model here — and why they are
fine-tunable in this pipeline too. `../jobs/ssl/` runs that grid.

**`-l<n>` truncates, it does not just select.** Layers past `n` are deleted at build time
rather than computed and discarded, so `wavlm-large-l4` is a 63.5M-parameter backbone instead
of a 315.5M one — the same budget as `eres2netv2-w24s4ep4`, and the reason fine-tuning these
is affordable at all. Outputs match the untruncated model to `atol=1e-5`. Layer 4 won for
every bundle and both targets in the weak sweep; the last layer was close to the worst.

**Padding is masked, which torchaudio's own path does not do for WavLM.**
`Encoder.extract_features` builds an additive `attention_mask`, and `WavLMSelfAttention`
asserts that argument is `None`; WavLM instead accepts a boolean `key_padding_mask` that
`Transformer.get_intermediate_outputs` never forwards. `SSLEncoder` writes out the layer loop
and passes whichever mask each attention implementation honours. Without it a row's embedding
moved by up to 6.1 absolute depending on what else was in its batch — which would have made
test predictions depend on batch composition. Verified: padded-batch vs single-row cosine is
1.000000 for `wavlm-large-l4` and `xlsr-300m-l4`, 0.999974 for `wavlm-base-plus-l4` (the
residual is the positional conv's 128-frame kernel reaching into the padding, which is
fairseq's behaviour too), and unpadded output matches `torchaudio`'s own path exactly.

Bundles pretrained on normalised audio (`WAVLM_LARGE`, `XLSR_300M`) are wrapped by torchaudio
in a `_Wav2Vec2Model` that layer-norms over the *whole batch tensor*, padding included. The
wrapper is unwrapped and the normalisation redone per row over valid samples — which is what
the batch-of-one path in `../weak/extract_features.py` effectively did.

## Interaction vector

`--interaction` selects how the two embeddings are combined before the head. With `d` the
embedding width (256 by default):

| mode | vector | dim | notes |
|---|---|---|---|
| `baseline` | `[a, b, |a−b|, a⊙b]` | 4d | the official baseline (default) |
| `scalars` | baseline + `[cos, ‖a−b‖]` | 4d+2 | similarity made explicit |
| `normed` | LayerNorm each block, then baseline | 4d | equalises block scales |
| `normed-scalars` | both of the above | 4d+2 | |
| `signed` | `[a, b, a−b, a⊙b]` | 4d | keeps the direction of the difference |
| `no-b` | `[a, |a−b|, a⊙b]` | 3d | drops the reference block |
| `symmetric` | `[a+b, |a−b|, a⊙b]` | 3d | `f(a,b) == f(b,a)` exactly |
| `bilinear` | baseline + `(Ua)⊙(Vb)` | 4d+r | `--bilinear-rank`, default 64 |
| `no-b-bilinear` | `no-b` + `(Ua)⊙(Vb)` | 3d+r | the two modes that gained most |

Only `normed*` and the two `bilinear` modes add parameters; the rest are free. `baseline`
adds no `state_dict` keys, so checkpoints written before this flag existed still load
unchanged.

`no-b-bilinear` combines the two modes that gained most in the first ablation (ECAPA + mlp
+ MSE, 20,000 steps, dev UTT-SRCC against the `baseline` control): `no-b` was +0.030 on
`spk_sim` and +0.002 on `acc_sim`, `bilinear` +0.017 and +0.030. They pull in compatible
directions — one removes the raw reference block, the other adds a learned comparison of
projected subspaces — so the combination keeps a route from `b` to the head while dropping
the 256 raw dimensions that measurably were not earning their place.

Two measurements motivate most of these, both on `coral_commonaccent-moe`, dev set:

- **The vector is role-dependent, not merely asymmetric.** Swapping the inputs moves
  SYS-SRCC from 0.932 to **−0.164**, and averaging `f(a,b)` with `f(b,a)` costs 0.10
  SYS-SRCC. That is not a defect to fix — `wav_a` is always the system output and `wav_b`
  always the `sys019` reference, in train, dev and test, so `f(b,a)` is never asked for.
  `symmetric` exists to ablate this, not because symmetry is expected to win.
- **The reference is used less than 1024 dimensions suggest.** Substituting a wrong
  reference from the same pool only drops SYS-SRCC from 0.932 to 0.786, so much of the
  system-level ranking comes from `a` alone — "how degraded is this sample" — rather than
  from any comparison. `no-b` tests how much the `b` block contributes at all.

`scalars` and `normed` follow from two properties of the current vector. The embeddings
are L2-normalised, so `sum(a⊙b)` *is* the cosine: the head has to discover a uniform sum
over `d` dimensions to recover a feature that scores SYS-SRCC 0.809 on its own zero-shot.
And the blocks arrive on very different scales — `a`, `b` on the unit sphere, `|a−b|` in
[0,2], entries of `a⊙b` around `1/d` — so the product block is attenuated before the first
`Linear` sees it.

Note that `cos` and `‖a−b‖` are monotone transforms of each other under L2 normalisation
(`‖a−b‖² = 2 − 2cos`), so `scalars` adds no information the vector lacked. The point is
that the head no longer has to find it.

Expect these to matter most with a frozen encoder: a fine-tuned 20M-parameter backbone can
compensate for a poor interaction. For a clean read, ablate with `--freeze-encoder` first,
which is minutes per run rather than hours.

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
| `--encoder` | `ecapa-voxceleb` | `python encoders.py --list`; combine with `+`; SSL names carry the layer, e.g. `wavlm-large-l4` |
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
