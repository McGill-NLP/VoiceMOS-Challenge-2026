# Rank-N-Contrast for Track 3

Pure [Rank-N-Contrast](https://arxiv.org/abs/2210.01189) (Zha et al., NeurIPS 2023) applied to
speaker/accent similarity prediction, compared head-to-head against the official baseline on the
same splits.

The reference implementation lives in [../papers/Rank-N-Contrast/](../papers/Rank-N-Contrast/); the
motivation and reading notes are in [../papers/IDEAS.md](../papers/IDEAS.md).

## The idea in one paragraph

The baseline trains end-to-end on the prediction and never constrains the representation. RNC instead
trains the encoder so that **distances in feature space are ordered by distances in label space**:
for each anchor, every other sample in the batch is a positive exactly once, and its negatives are
everything *farther from the anchor in label space*. The encoder is then frozen and a head is fitted
on top. The paper's argument for why this should matter here: its Fig. 6 shows L1-trained features
clustering by *which webcam took the photo* rather than by temperature — our analogue is features
clustering by `system_id` rather than by similarity — and its Fig. 5 shows the gain *growing* as the
training set shrinks, which is our regime at 2,800 unique pairs.

## What a "sample" is here

In the paper one image → one age. Here one **(wav_a, wav_b) pair** → one score, so the RNC feature is
a *pair* representation and RNC contrasts pairs against pairs:

```
ECAPA-TDNN → Linear(192, 256) → L2 norm ─┐
                                         ├→ v = [e_a, e_b, |e_a−e_b|, e_a⊙e_b]   (1024-d)
ECAPA-TDNN → Linear(192, 256) → L2 norm ─┘
```

`v` is exactly the baseline's interaction vector. Per the paper there is **no separate projection
head** for the contrastive loss — RNC is applied to the same feature the regression head consumes.
Labels are the per-pair mean over the ~5 listener ratings, matching the baseline's aggregation.

## ⚠️ The baseline does not fine-tune ECAPA

Worth knowing before interpreting any comparison. The baseline's `finetune.py` builds its model with
`freeze_ssl=False  # Fine-tuning everything`, but SpeechBrain's `Pretrained.__init__` defaults to
`freeze_params=True`, which sets `requires_grad=False` on all 22.15M ECAPA parameters. Verified:

```
baseline Model(freeze_ssl=False): total 22,265,985  trainable 115,073  (0.52%)
  ECAPA submodule: 22,150,912 params, 0 trainable
```

So **baseline 2 trains only the 192→256 projection and the MLP head** — 0.52% of the model. A second
consequence: SpeechBrain's freeze also calls `.eval()`, but the later `model.train()` puts ECAPA back
into training mode, so its BatchNorm *running statistics* keep drifting even though its weights are
frozen.

This directory reproduces that behaviour by default so the comparison is honest, and exposes both
knobs explicitly:

| flag | effect | default |
|---|---|---|
| `--unfreeze-ecapa` | genuinely train the 22.15M ECAPA parameters | off (= baseline) |
| `--ecapa-eval-mode` | pin ECAPA to `eval()`, stopping BatchNorm drift | off (= baseline) |

Both training scripts log the trainable/total parameter count on startup, so this can never be
silently wrong again.

## Data

Uses the official distribution under [../baseline/data/](../baseline/data/), exactly as the baseline
does — no derived splits:

| file | rows | contents |
|---|---|---|
| `sets/train.csv` | 13,687 | listener-wise ratings: 21 systems, 25 listeners, 2,800 unique pairs |
| `sets/dev.csv` | 600 | **unlabelled** — `system_id,utterance_id,wav_a_path,wav_b_path` only |

All scripts aggregate the listener-wise rows to per-pair means, matching the baseline.

### There is no local validation signal

`sets/dev.csv` ships without scores during the training phase, so nothing here can be scored offline.
Consequences, all matching what the baseline does:

* **No checkpoint selection.** Both stages train for a fixed number of steps and keep the last
  checkpoint, exactly as the baseline's fixed 20,000 steps do. `--val-csv` exists in both scripts and
  turns selection back on the moment a labelled split is available.
* **Comparison against the baseline goes through CodaBench.** Produce a prediction CSV for
  `sets/dev.csv` and submit it; the baseline's own dev numbers are in
  [../baseline/README.md](../baseline/README.md) (baseline 2, spk_sim: UTT-SRCC 0.451, SYS-SRCC
  0.860; acc_sim: UTT-SRCC 0.440, SYS-SRCC 0.861).
* `calculate_metrics.py` is for when dev labels are released, or for any labelled split you pass it.
  It also takes `--held-out-from TRAIN_CSV` to report metrics restricted to audio pairs absent from
  training, which is worth using with any split built at the `(system_id, listener_id)` level — such
  splits share audio pairs with train and the overall number blends memorization with generalization.

## Running

### On the cluster (recommended)

Two Slurm scripts in [../jobs/](../jobs/) run the whole thing unattended, each sweeping two learning
rates (1e-3, the baseline's value, and 1e-4) and producing CodaBench-ready dev predictions:

```bash
sbatch track3/jobs/vmc26-t3-rnc-spk.sh     # speaker similarity
sbatch track3/jobs/vmc26-t3-rnc-acc.sh     # accent similarity
```

One L40S, 8 h wall clock (~3 h of work: 1.4 h per learning rate for stage 1, plus a couple of minutes
each for stage 2 and inference). Outputs land in `egs/<metric>_lr<lr>/`, and the job log ends with the
paths of the submission CSVs.

**Batch size is capped at 128.** At 256, ECAPA's attentive-pooling `F.pad` raises `input tensor must
fit into 32-bit index math` — not an OOM but a hard indexing limit, triggered because repetitive
padding stretches every clip in a batch to the longest one. Measured at 128: 19.85 GiB peak,
0.56 s/step, 21.9 steps/epoch. Going above 128 needs `--max-audio-sec` to cap the padded length.

Stage 1 checkpoints every 1,750 steps (~100 epochs). Since stage 2 costs about two minutes, once dev
labels are released you can re-probe several stage-1 checkpoints and pick by real dev SRCC instead of
committing to the last one.

### Interactively

```bash
module load miniconda/3 && conda activate VoiceMOS

DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
```

### Stage 1 — RNC representation learning

```bash
python train_rnc.py \
    --data-root $DR --train-csv $DR/sets/train.csv \
    --target-metric spk_sim --outdir egs/rnc_spk_sim \
    --batch-size 64 --train-steps 13000
```

No head, no regression loss. Logs the RNC loss, its lower bound `L*`, and the gap between them;
writes `encoder_last.pt`.

### Stage 2 — linear probe on the frozen encoder

```bash
python train_head.py \
    --data-root $DR --train-csv $DR/sets/train.csv \
    --target-metric spk_sim --outdir egs/rnc_spk_sim/head \
    --encoder-ckpt egs/rnc_spk_sim/encoder_last.pt --freeze-encoder \
    --head linear --loss l1
```

Because a frozen encoder gives deterministic features, they are extracted once and cached; the probe
then trains in seconds.

### Baseline arm (for comparison, same data code)

```bash
python train_head.py \
    --data-root $DR --train-csv $DR/sets/train.csv \
    --target-metric spk_sim --outdir egs/baseline_spk_sim \
    --head mlp --loss mse --batch-size 16 --lr 1e-3 --train-steps 20000
```

No `--encoder-ckpt` and no `--freeze-encoder` → end-to-end training from pretrained ECAPA with the
MLP head and range clipping, i.e. official baseline 2, but driven through identical data handling.

### Predict on the official dev set (for CodaBench)

```bash
python inference.py --data-root $DR --csv-path $DR/sets/dev.csv \
    --checkpoint egs/rnc_spk_sim/head/model_last.pt --target-metric spk_sim \
    --out egs/rnc_spk_sim/head/dev.csv
```

`inference.py` handles unlabelled CSVs: the `spk_sim` column comes out empty and no metrics are
printed. Once dev labels are released:

```bash
python calculate_metrics.py --prediction-csv egs/rnc_spk_sim/head/dev.csv \
    --ground-truth-csv <labelled dev csv>
```

Repeat any of the above with `--target-metric acc_sim` for the accent model.

## Files

| file | contents |
|---|---|
| `loss.py` | RNC loss, the paper's `L*` lower bound, the feature/label rank-correlation diagnostic |
| `model.py` | `SpeechEncoder` / `PairEncoder` / heads, with the freezing behaviour documented |
| `data.py` | pair dataset, per-pair mean aggregation, repetitive padding, reference-side dedup |
| `train_rnc.py` | stage 1 |
| `train_head.py` | stage 2 and the baseline arm |
| `inference.py`, `calculate_metrics.py`, `metrics.py` | prediction and scoring |
| `test_loss.py` | correctness tests — run `python test_loss.py` |

### Notes on the loss implementation

`loss.py` carries two implementations. `rnc_loss_reference` is a line-for-line port of the authors'
code, used only for testing. `rnc_loss` is the default and is what training uses: it sorts each row by
label distance, which turns every denominator into a suffix sum, eliminating the authors' Python loop
over positives and their `[n, n, d]` intermediate. Measured on an L40S at d=1024:

| batch | vectorised | reference | speedup | peak memory |
|---|---|---|---|---|
| 256 | 1.1 ms | 39.0 ms | 35× | 25 vs 84 MiB |
| 512 | 1.3 ms | 82.3 ms | 62× | 37 vs 539 MiB |

This matters because the paper's Table 6a shows accuracy increasing monotonically with the number of
in-batch positives, so we want the largest batch that fits.

`test_loss.py` checks the two implementations agree to 1e-5 across four label regimes (distinct,
tied integers, means-of-5-ratings, multi-dimensional) × three similarity measures × three batch
sizes, and verifies the paper's theorems: `L_RNC > L*` (Theorem 1), `L*` matching its closed form
under full ties, and label-ordered features driving the loss to `L*` while shuffled features do not
(Theorem 3).

**Read `L_RNC − L*`, not `L_RNC`.** `L*` is the tight lower bound and depends on the *tie structure*
of the batch's labels. Our labels are means of ~5 integer ratings, so ties are common and `L*` sits
well above zero (≈5.5 for a batch of 256). Raw loss values are not comparable across batches.

## Defaults and where they come from

| setting | value | source |
|---|---|---|
| feature similarity | negative L2, unnormalised | paper Table 6b (cosine 6.51 vs neg-L2 6.14 MAE) |
| temperature | 2.0 | paper's grid search over {0.1, 0.2, 0.5, 1, 2, 5} |
| positives | all in-batch | paper Table 6a (monotone in K) |
| training scheme | two-stage, frozen encoder | paper Table 6c (probing 6.14 < fine-tuning 6.36 < joint 6.42) |
| views | one | paper Appendix G.3 — augmentation is not essential for RNC |
| stage-1 LR | 1e-3, AdamW | the baseline's value, kept for parity |
| stage-1 schedule | cosine annealing | paper |

Two of these deserve scepticism. **The learning rate is the baseline's, not a tuned one** — the paper
grid-searched LR per dataset and used SGD on a from-scratch ResNet, which does not transfer to a
pretrained encoder; this is the first knob to tune. And **one view** is chosen deliberately: every
obvious audio augmentation (speed, pitch, VTLP) changes the voice, i.e. changes the very label being
regressed, so two-view augmentation is a correctness risk here, not just a tuning choice. Random
temporal cropping (`--max-audio-sec`) is the safe one to add first.
