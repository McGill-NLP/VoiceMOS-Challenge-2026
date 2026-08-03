# Track 3 — the UTMOS objective

Tests whether UTMOS's loss (Saeki et al., *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge
2022*, Interspeech 2022 — paper in [../papers/](../papers/), code in
[../papers/UTMOS22](../papers/UTMOS22)) improves over the Track 3 baseline.

**Only the objective changes.** Same SpeechBrain ECAPA encoder, same 256-d projection, same 4-way
interaction vector, same range-clipped MLP head, same data pipeline as
[../baseline/](../baseline/). Any difference in the numbers is attributable to the loss.

## The losses

UTMOS's final objective (paper Eq. 1) is

```
L = beta * L_reg + gamma * L_con
```

**Contrastive loss** — over every ordered pair *(i, j)* in the batch:

```
L_con(i,j) = max(0, |d_ij - d̂_ij| - margin),    d_ij = y_i - y_j,   d̂_ij = ŷ_i - ŷ_j
```

It asks the model to reproduce *score differences*, not scores. That penalises getting the sign of
a comparison wrong, which is exactly what SRCC measures — and MSE is indifferent to. Concretely,
for predictions that are a constant offset from the truth (rank perfectly preserved):

| | contrastive | MSE |
|---|---|---|
| constant offset, rank preserved | **0.000** | 1.000 |
| rank inverted | **0.889** | 3.667 |

**Clipped MSE** — `1(|y - ŷ| > tau) * (y - ŷ)²`, a dead zone around the target. A rating of 4 means
"about a 4", so driving the prediction to exactly 4.0 overfits the label.

Both are faithful ports of `strong/loss_function.py` in
[UTMOS22](https://github.com/sarulab-speech/UTMOS22), verified against the upstream
implementation over 2,400 randomised comparisons (0 mismatches, `atol=1e-7`). Only the frame-level
machinery is dropped: UTMOS scores every frame and averages at inference, whereas Track 3 emits one
scalar per pair.

Hyperparameters are UTMOS's shipped values from `strong/configs/train/default.yaml`:
`beta=1.0, gamma=0.5, tau=0.25, margin=0.1`. (The code default for `margin` is 0.2; the config
overrides it to 0.1.)

## The ablation

`--loss` selects the arm:

| Arm | Objective | Purpose |
|---|---|---|
| `mse` | plain MSE | the control — what the baseline trains on |
| `clipped` | clipped MSE alone | isolates the dead zone |
| `contrastive` | contrastive alone | UTMOS Table 2a "w/o MSE loss" shows this is viable on its own |
| `utmos` | `1.0 * clipped + 0.5 * contrastive` | the paper's Eq. 1 |

```bash
sbatch track3/jobs/voicemos-track3-utmos-loss.sh
```

runs all arms (plus `utmos-g2`, `gamma=2.0`) × `{spk_sim, acc_sim}`, scoring the labelled dev set
throughout training and writing submission CSVs from the best checkpoint. Standalone:

```bash
python finetune.py --data-root $DR --train-csv $DR/sets/train.csv \
    --target-metric spk_sim --loss utmos --outdir egs/run \
    --batch-size 16 --encoder-lr 1e-5 --lr 1e-3 \
    --dev-csv $EVAL_DR/sets/dev_with_labels.csv --dev-data-root $DR --eval-steps 250
```

## Two things that will bite you

**Batch size feeds the contrastive term quadratically.** A batch of *B* contributes *B(B−1)* ordered
pairs: 16 → 240, 32 → 992, 48 → 2256. **Gradient accumulation does not substitute** — the loss is
computed per micro-batch, so accumulation just averages several small pair sets. Measured on an
L40S (46 GB), ECAPA full fine-tuning with `--loss utmos`:

| Batch | Pairs/step | s/step | Peak GPU | 4k steps |
|---|---|---|---|---|
| 16 | 240 | 0.158 | 9.98 GiB | 10.5 min |
| 32 | 992 | 0.314 | 19.69 GiB | 20.9 min |
| 48 | 2256 | 0.514 | 31.77 GiB | 34.3 min |
| 64 | 4032 | 0.667 | 42.31 GiB | 44.4 min |

The job script holds batch at 16 so the comparison against the baseline is clean; re-run with
`--export=ALL,BATCH=32` to test whether more pairs help.

**`contrastive` alone fixes no absolute scale.** It constrains only differences, so MSE will be bad
even when the correlations are excellent — in the smoke test, `srcc_sys` 0.81 with `mse_utt` 2.5.
Range clipping (`tanh*2+3`) bounds predictions to [1,5] but does not calibrate them. If that arm
wins on correlation, recover MSE with a post-hoc monotone fit on dev (an affine rescale, or
isotonic regression) before submitting — it cannot hurt SRCC, which is rank-invariant.

## The encoder is genuinely fine-tuned here

[../baseline/model.py](../baseline/model.py) passes `freeze_ssl=False  # Fine-tuning everything`,
but `EncoderClassifier.from_hparams` never overrides SpeechBrain's `Pretrained(freeze_params=True)`
default, which sets `requires_grad=False` across the backbone. The published Baseline 2 is a
**frozen ECAPA with a trained 0.12M head** (verified: 22.27M total, 0.12M trainable, 0.00M inside
ECAPA).

This package passes `freeze_params=False` and makes freezing explicit via `--freeze-encoder`.
Consequently the baseline's `--lr 1e-3` is wrong here — it was tuned for a 0.12M head. Default is
backbone `1e-5` / head `1e-3`.

Baseline-format checkpoints still load: the backbone key prefix is remapped and the head name is
recovered from the state dict (verified 0 missing / 0 unexpected against
`../baseline/official-egs/spk_sim_adamw_lr1e-3/model_spk_sim_step20000.pt`).

## Early signal

From a 120-step smoke run on `spk_sim` (far too short to conclude anything, but the gap is large):

| Arm | `srcc_sys` | `srcc_utt` | `mse_utt` |
|---|---|---|---|
| `mse` | 0.621 | 0.287 | 0.526 |
| `clipped` | 0.595 | 0.281 | 0.530 |
| **`contrastive`** | **0.813** | **0.448** | 2.515 |
| `utmos` (γ=0.5) | 0.606 | 0.288 | 0.526 |

The contrastive term converges on *rank* far faster than the regression terms, while the combined
loss at UTMOS's shipped `gamma=0.5` tracks plain MSE closely — the regression term appears to
dominate early on this dataset. That is why the job script includes a `gamma=2.0` arm. Treat these
as a hypothesis to test at full length, not a result.

## Layout

```
losses.py              ContrastiveLoss, ClippedMSELoss, CombineLosses, build_loss(). Ported from UTMOS22.
model.py               ECAPA + projection + interaction + range-clipped head. Same as the baseline.
finetune.py            Training with --loss, plus dev scoring during training.
inference.py           Checkpoint-driven encoder/metric resolution.
calculate_metrics.py   Unchanged copy of ../baseline/calculate_metrics.py.
make_eval_gt.py        Averages a listener-wise split into one row per pair.
```

## What is deliberately not here

UTMOS's other components, in rough order of expected value for this task:

- **Listener-dependent modelling with a mean listener.** The single biggest ablation effect in the
  paper (OOD `srcc_sys` 0.972 → 0.944 without it). Track 3 gives 25 listeners with ~550 ratings
  each, which is far better conditioned than UTMOS's 288 listeners; the baseline throws this away by
  averaging per pair. See [../papers/IDEAS.md](../papers/IDEAS.md).
- **Frame-level scoring** — needs rethinking for a pair task.
- **Phoneme encoding**, requiring an ASR pass.
- **Stacking strong and weak learners**, which is where much of UTMOS's final margin came from.
