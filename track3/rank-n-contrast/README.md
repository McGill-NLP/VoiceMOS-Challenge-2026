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

**Stage 2 reproduces that behaviour by default so the comparison is honest. Stage 1 does not, and
must not** — see below.

| flag | script | effect | default |
|---|---|---|---|
| `--freeze-ecapa` | `train_rnc.py` | freeze the backbone, leaving RNC only the 49k projection | off → **ECAPA trains** |
| `--unfreeze-ecapa` | `train_head.py` | genuinely train the 22.15M ECAPA parameters | off (= baseline) |
| `--ecapa-eval-mode` | both | pin ECAPA to `eval()`, stopping BatchNorm drift | off (= baseline) |

Both training scripts log the trainable/total parameter count on startup, so this can never be
silently wrong again.

### Why stage 1 must fine-tune the backbone

The first sweep froze ECAPA in stage 1, and the result was a stage that did essentially nothing:

```
Trainable parameters: 49,408 / 22,200,320 (0.22%) | ECAPA FROZEN
step     50/8750 | loss 4.0010 | batch L* 2.3863 | gap +1.6147
step   8750/8750 | loss 3.9297 | batch L* 2.3480 | gap +1.5816
```

0.07 nats over 8,750 steps and ~100 GPU-minutes. With the backbone fixed, the contrastive objective
can only rotate a single `Linear(192, 256)` on top of frozen features — there is no representation
left for it to learn. Scored against the released dev labels, those runs land below baseline 2 on
both targets (best spk_sim 0.410 UTT-SRCC / 0.808 SYS-SRCC; best acc_sim 0.437 / 0.677).

This was not merely a default: `--unfreeze-ecapa` **crashed on contact**. `SpeechEncoder.__init__`
probes the encoder output dim with `encode_batch(torch.zeros(1, 16000))`, and SpeechBrain only calls
`.eval()` itself when `freeze_params=True`. Unfrozen, that batch of 1 reached a `BatchNorm1d` in
train mode and raised `Expected more than 1 value per channel ... torch.Size([1, 6144, 1])` before
training began. The probe now runs under a temporary `.eval()`, so both modes construct.

Stage 2 keeps its frozen default: the RNC arm is a linear probe on frozen features (the paper's best
variant, Table 6c), and the baseline arm needs the frozen behaviour to reproduce baseline 2.

## Data

Uses the official distribution under [../baseline/data/](../baseline/data/), exactly as the baseline
does — no derived splits:

| file | rows | contents |
|---|---|---|
| `sets/train.csv` | 13,687 | listener-wise ratings: 21 systems, 25 listeners, 2,800 unique pairs |
| `sets/dev.csv` | 600 | **unlabelled** — `system_id,utterance_id,wav_a_path,wav_b_path` only |
| `../vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv` | 600 | the same pairs **with** `spk_sim` and `acc_sim` |

All scripts aggregate the listener-wise rows to per-pair means, matching the baseline.

### The labelled dev set

`dev_with_labels.csv` was released after the first sweep ran, and the Slurm scripts now use it
throughout. Pass it as `--val-csv` to either stage.

**Use `--data-root` = the *train*-phase distro even for this CSV.** It ships in the eval-phase
distro, but all 748 dev waveforms are present under the train-phase root while the eval-phase root is
missing 160 of them (588/748). The scripts already resolve it this way, as do the encoder jobs.

What it turns on:

* **Stage 1 monitoring.** `--val-csv` reports the RNC loss against its lower bound `L*` and the
  feature/label rank correlation (the paper's Table 1 diagnostic) every `--eval-steps`, and saves
  `encoder_best.pt` by that correlation.
* **Stage 2 checkpoint selection.** `--select-on sys_srcc` keeps `model_best.pt` at the best dev
  system-level SRCC instead of taking the final step blind.
* **Stage-1 checkpoint selection.** Probing costs about a minute on cached features, so the job
  scripts probe *every* stage-1 checkpoint and keep the one with the best dev SYS-SRCC. A
  contrastive stage has no reason to peak exactly at the last step, and stage 1's own
  `encoder_best.pt` criterion does not always agree with downstream dev SRCC.

⚠️ **Dev numbers are now selection-contaminated.** The same 600 pairs choose the stage-1 checkpoint
and the head step, so the metrics the job prints are optimistic and are only useful for ranking runs
against each other. The honest comparison is still the CodaBench eval set. Baseline 2's published dev
numbers, for reference: spk_sim UTT-SRCC 0.451 / SYS-SRCC 0.860; acc_sim 0.440 / 0.861
([../baseline/README.md](../baseline/README.md)).

`calculate_metrics.py` scores any labelled split. It also takes `--held-out-from TRAIN_CSV` to
restrict metrics to audio pairs absent from training, worth using with any split built at the
`(system_id, listener_id)` level — such splits share audio pairs with train, so the overall number
blends memorization with generalization.

## Running

### On the cluster (recommended)

Two Slurm scripts in [../jobs/](../jobs/) run the whole thing unattended, each sweeping two learning
rates and producing CodaBench-ready dev predictions:

```bash
sbatch track3/jobs/voicemos-track3-rank-n-contrast-speaker.sh   # speaker similarity
sbatch track3/jobs/voicemos-track3-rank-n-contrast-accent.sh    # accent similarity
```

One L40S, 12 h wall clock (~6 h of work: 2.7 h per learning rate for stage 1, ~8 min to probe its
five checkpoints, plus a couple of minutes for inference). Outputs land in `egs/<metric>_ftlr<lr>/` —
`ftlr`, not `lr`, so they do not overwrite the frozen-ECAPA sweep already in `egs/`. The submission
CSV is `egs/<metric>_ftlr<lr>/dev_<metric>.csv`, matching the layout in `../encoders/egs/`, and each
run's per-checkpoint probes are kept in `probe_encoder_*/` alongside it. The job log ends with the
submission paths and prints dev metrics per run (optimistic — see the selection caveat above).

**Learning rates are 1e-4 and 1e-5.** The frozen sweep used (1e-3, 1e-4), calibrated for a 49k
projection; those rates applied to a 22.15M pretrained backbone would destroy the features it starts
from.

**Batch size is 96 with `--max-audio-sec 6`.** Training ECAPA rather than just the projection costs
roughly 5× the memory, so the frozen sweep's 128 no longer fits. Measured on a 46 GiB L40S with the
backbone trainable:

| batch | crop | peak | s/step |
|---|---|---|---|
| 64 | — | 36.6 GiB | 0.80 |
| 64 | 6 s | 22.3 GiB | 0.58 |
| 96 | 6 s | 31.6 GiB | 0.84 |
| 128 | 6 s | 39.9 GiB | 1.04 |

Without a crop, peak memory follows the *longest* clip in the batch — repetitive padding stretches
every clip to it — and clips reach 14.9 s against a 4.8 s mean, so an unlucky batch OOMs hours into a
run. The 6 s crop bounds that; p95 duration is 6.9 s, so most clips are untouched and the random crop
doubles as augmentation over only 2,800 unique pairs. RNC gains monotonically with in-batch positives
(paper Table 6a), so 96 is the largest batch that still leaves real headroom; 128 fits if you can
accept 6 GiB of margin. The crop applies to stage 1 only — stage 2 and inference use full audio.

Stage 1 checkpoints every 2,340 steps (~80 epochs), and each checkpoint is probed against the
labelled dev set so the kept representation is chosen rather than assumed.

### Interactively

```bash
module load miniconda/3 && conda activate VoiceMOS

DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
DEV_LABELS=../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv
```

### Stage 1 — RNC representation learning

```bash
python train_rnc.py \
    --data-root $DR --train-csv $DR/sets/train.csv \
    --target-metric spk_sim --outdir egs/rnc_spk_sim \
    --batch-size 96 --max-audio-sec 6 --lr 1e-5 --train-steps 11700 \
    --val-csv $DEV_LABELS --eval-steps 500
```

All 22.15M ECAPA parameters train. Add `--freeze-ecapa` for the ablation that keeps them fixed.

No head, no regression loss. Logs the RNC loss, its lower bound `L*`, and the gap between them;
writes `encoder_last.pt`, plus `encoder_best.pt` by feature/label rank correlation when `--val-csv`
is given.

### Stage 2 — linear probe on the frozen encoder

```bash
python train_head.py \
    --data-root $DR --train-csv $DR/sets/train.csv \
    --target-metric spk_sim --outdir egs/rnc_spk_sim/head \
    --encoder-ckpt egs/rnc_spk_sim/encoder_last.pt --freeze-encoder \
    --head linear --loss l1 \
    --val-csv $DEV_LABELS --select-on sys_srcc
```

Because a frozen encoder gives deterministic features, they are extracted once and cached; the probe
then trains in seconds. Repeat it against each `encoder_step*.pt` and keep the best `model_best.pt` —
that is exactly what the job scripts automate.

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
    --checkpoint egs/rnc_spk_sim/head/model_best.pt --target-metric spk_sim \
    --out egs/rnc_spk_sim/dev_spk_sim.csv
```

`inference.py` handles unlabelled CSVs: the `spk_sim` column comes out empty and no metrics are
printed. Score the result against the labelled dev set with:

```bash
python calculate_metrics.py --prediction-csv egs/rnc_spk_sim/dev_spk_sim.csv \
    --ground-truth-csv $DEV_LABELS
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
