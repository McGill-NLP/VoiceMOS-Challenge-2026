# Frozen-feature weak learners and stacking (UTMOS22's other half)

Classical regressors — ridge, SVR, random forest, gradient boosting — fitted on **frozen**
encoder embeddings, then pooled with the fine-tuned deep models from
[../unified/](../unified/).

This is the *stacking* half of UTMOS22 (Saeki et al., Interspeech 2022). The *loss* half of
the same paper is a separate experiment in [../utmos-approach/](../utmos-approach/); the two
are independent and can be read in either order.

Nothing here is fine-tuned. Every encoder runs once, under `torch.inference_mode()`, and the
embeddings are cached to disk. All the learning happens in scikit-learn on the CPU.

**Result: `spk_sim` dev UTT-SRCC 0.609 and `acc_sim` 0.577**, against 0.579 / 0.564 for the
deep pool alone — and 0.607 / 0.579 for the best previous system. On `spk_sim` this is the
first ensemble gain in the project whose 95% CI excludes zero: **+0.029 [+0.012, +0.047]**.

---

## Why this experiment exists

Not to build a better model — to build a *differently wrong* one.

Ensemble gain on this dev set tracks the mean pairwise correlation of member **residuals**
almost linearly:

| pool | mean residual corr | gain over mean member |
|---|---|---|
| heterogeneous 16 (mixed recipes) | 0.878 | +0.062 |
| factorial 16 (one fixed recipe) | 0.921 | +0.047 |
| factorial top-8 | 0.957 | +0.019 |

The deep pool was stuck near 0.92, and adding a fourth speaker-ID encoder (`ecapa-voxceleb`,
completing a 4x2x2 factorial) did not move it. That failure is the motivation here: all four
registry encoders are discriminative speaker/accent-ID nets trained with AAM-softmax and
attentive-stats pooling, so they share an inductive bias — they are trained to be *invariant*
to the channel and phonetic variation that accent lives in. Swapping one for another changes
little about *how* they are wrong.

Two departures were available:

1. **Change the function class.** A ridge regression on frozen features has no gradient path
   into the encoder and no shared optimisation noise with a fine-tuned network.
2. **Change the pretraining objective.** Masked-prediction SSL models (WavLM, wav2vec 2.0,
   XLS-R) *retain* the phonetic and prosodic detail speaker-ID encoders discard.

This directory does both.

---

## Encoders

Thirteen feature sets from seven encoders. Speaker/accent-ID models are loaded through
[../unified/encoders.py](../unified/encoders.py); SSL models through `torchaudio.pipelines`,
which ships every backbone UTMOS22 used — **`fairseq` is not required**, so
`../papers/UTMOS22/fairseq_checkpoints/download_*.sh` can be ignored.

| spec | dim | pretraining | reference |
|---|---|---|---|
| `sv:ecapa-voxceleb` | 192 | speaker ID, VoxCeleb, AAM-softmax | ECAPA-TDNN, Desplanques et al. 2020 |
| `sv:commonaccent-ecapa` | 192 | accent ID, CommonVoice | CommonAccent, Zuluaga-Gomez et al. 2023 |
| `sv:eres2netv2` | 192 | speaker ID, 3D-Speaker | ERes2NetV2, Chen et al. 2024 |
| `sv:eres2netv2-w24s4ep4` | 192 | speaker ID, 53.5M params | ERes2NetV2, Chen et al. 2024 |
| `ssl:WAVLM_BASE_PLUS` | 768 | masked prediction + denoising | WavLM, Chen et al. 2022 |
| `ssl:WAVLM_LARGE` | 1024 | masked prediction + denoising | WavLM, Chen et al. 2022 |
| `ssl:WAV2VEC2_XLSR_300M` | 1024 | multilingual, 128 languages | XLS-R, Babu et al. 2021 |

SSL bundles are mean-pooled over time, which is exactly what UTMOS22's
`extract_ssl_feature.py` does (`res[0].squeeze(0).mean(dim=0)`).

**Layers.** One forward pass returns every transformer layer, so extra layers are free.
Layers 4, 8 and last are kept, giving 3 files per SSL bundle. UTMOS22 used only the last
layer; see the findings below for why that turns out to be close to the worst choice here.

**Resampling matters.** The corpus is 24 kHz and every encoder expects 16 kHz. Skipping the
resample does not raise an error — it silently feeds audio at 1.5x speed and produces quietly
wrong embeddings. [extract_features.py](extract_features.py) resamples, matching
[../unified/finetune.py](../unified/finetune.py).

---

## Features

The interaction algebra is the baseline's, so that a weak learner and a deep model see the
same information and any difference in their errors comes from the function class:

```
full      [e_a, e_b, |e_a - e_b|, e_a * e_b, cos, ||e_a - e_b||]    4D + 2
compact   [           |e_a - e_b|, e_a * e_b, cos, ||e_a - e_b||]   2D + 2
```

`compact` drops the raw embedding blocks. Motivated by the interaction ablation in
[../unified/interactions.py](../unified/interactions.py): substituting a *wrong* sys019
reference only moved SYS-SRCC 0.932 -> 0.786, so much of the signal is in `e_a` alone.

Embeddings wider than 256 are PCA-reduced to 128 first, fitted on **train only**. Without it,
1024-d SSL embeddings give 4,098 `full` features against 2,800 training pairs. Feature
matrices are standardised on train; SVR and ridge are both scale-sensitive.

---

## Learners

The five families from UTMOS22's
[`models.py`](../papers/UTMOS22/stacking/ensemble_multidomain_scripts/models.py):

| name | estimator | grid |
|---|---|---|
| `ridge` | `sklearn.linear_model.Ridge` | alpha in {1, 10, 100, 1e3, 1e4} |
| `linsvr` | `sklearn.svm.LinearSVR` | C in {0.01, 0.1, 1} |
| `ksvr` | `sklearn.svm.SVR` (rbf) | C in {1, 10} x gamma in {scale, 1e-3} |
| `rf` | `RandomForestRegressor` | 300 trees, depth in {None, 12}, leaf in {1, 5} |
| `hgb` | `HistGradientBoostingRegressor` | 300 iters, lr in {0.05, 0.1}, leaves in {15, 31} |

**Two substitutions from the original recipe**, both forced by the environment and both
noted so the comparison stays honest:

- `HistGradientBoostingRegressor` replaces **LightGBM** (not installed). Same algorithm
  family — histogram-binned gradient boosting — no new dependency.
- A small fixed grid replaces **Optuna** (not installed). Deliberately small: with 2,800
  pairs and a +/-0.05 measurement floor on dev, an exhaustive search fits noise.

13 feature files x 2 feature sets x 2 targets x 5 learners = **260 models**, 6.0 CPU-hours
total (21,673 s of fitting).

---

## Protocol

**Grouped CV, not random folds.** Hyperparameters are chosen by `GroupKFold(5)` on
`system_id`. dev contains two systems absent from train (`sys003`, `sys015`) and test contains
four (`sys003`, `sys004`, `sys015`, `sys021`) — 344 of 600 test pairs come from unseen
systems. Random folds would let a model memorise system identity and pick hyperparameters that
do not survive the real split. UTMOS22 used plain CV; that is not safe here.

**dev is for scoring only.** Weak learners are fitted on the 2,800 train pairs. Every reported
number comes from `dev_with_labels.csv`. `test.csv` is unlabelled — predictions are written
for submission packaging and are never scored locally.

**Targets** are the per-pair mean over raters (train has ~5 rating rows per pair). Duplicating
identical feature rows would only reweight pairs by rater count.

---

## Results

All on `dev_with_labels.csv`, 600 pairs.

### Best single weak models

| target | encoder | features | learner | uSRCC |
|---|---|---|---|---|
| `spk_sim` | `eres2netv2` | full | ksvr | **0.534** |
| `spk_sim` | `eres2netv2` | full | ridge | 0.527 |
| `spk_sim` | `WAV2VEC2_XLSR_300M_l4` | full | ridge | 0.516 |
| `acc_sim` | `WAV2VEC2_XLSR_300M_l4` | full | hgb | **0.512** |
| `acc_sim` | `ecapa-voxceleb` | full | ridge | 0.501 |
| `acc_sim` | `commonaccent-ecapa` | full | hgb | 0.499 |

### Pools

| | uMSE | uLCC | uSRCC | sMSE | sLCC | sSRCC |
|---|---|---|---|---|---|---|
| **`spk_sim`** | | | | | | |
| deep top-8 | 0.349 | 0.621 | 0.579 | 0.049 | 0.938 | 0.907 |
| weak top-8 | 0.341 | 0.635 | 0.602 | 0.048 | **0.953** | **0.921** |
| **deep + weak** | **0.336** | **0.642** | **0.609** | **0.047** | 0.950 | 0.919 |
| **`acc_sim`** | | | | | | |
| deep top-8 | 0.316 | 0.601 | 0.564 | 0.039 | **0.924** | **0.933** |
| weak top-8 | 0.312 | 0.609 | 0.561 | 0.039 | 0.906 | 0.888 |
| **deep + weak** | **0.305** | **0.623** | **0.577** | **0.037** | 0.922 | 0.913 |

Gain over deep-only, bootstrap 95% CI: `spk_sim` **+0.029 [+0.012, +0.047]**,
`acc_sim` +0.013 [-0.006, +0.029].

### Decorrelation — the thing the experiment was actually testing

| | `spk_sim` | `acc_sim` |
|---|---|---|
| within deep top-8 | 0.957 | 0.927 |
| within weak top-8 | 0.851 | 0.867 |
| deep vs weak, cross-pool | 0.866 | 0.859 |

The weak pool is internally more diverse than the deep pool, and the cross-pool figure sits
below the 0.878 that the heterogeneous pool needed to earn +0.062. The mechanism worked.

---

## Findings

**SSL layer 4 beats layer 24, decisively.** For every bundle the early layer wins:
`XLSR_300M` scores 0.516 at layer 4 vs 0.438 at layer 24 (`spk_sim`), and 0.512 vs 0.381
(`acc_sim`). UTMOS22 used only the last layer. Last layers of masked-prediction models are
specialised toward the pretraining objective; the speaker- and channel-bearing information
that similarity assessment needs sits lower.

**`WAV2VEC2_XLSR_300M_l4` is the best `acc_sim` encoder in the sweep** (0.512), ahead of
`commonaccent-ecapa` (0.499) — the purpose-built accent encoder. A multilingual SSL model
treats accent variation as signal; an accent *classifier* compresses it to 16 logits.

**Weak learners are not weak.** On `spk_sim` the weak pool alone (0.602) beats the deep pool
alone (0.579), and an RBF-SVR on frozen `eres2netv2` features (0.534) matches fine-tuned
`ecapa-voxceleb` models (0.479-0.521) at seconds of CPU each. This is evidence that gradient
fine-tuning contributes less on 2,800 pairs than assumed.

**`full` beats `compact` everywhere** — 0.534 vs 0.498 (`spk_sim`), 0.512 vs 0.491
(`acc_sim`). The raw embedding blocks earn their place for a classical regressor even though
the `no-b` ablation favoured dropping them for a deep head.

**Learned stacking weights buy nothing over a plain average.** On `spk_sim` the unweighted
mean *won*, with `alpha`, `nnls` and `ridge` all within 0.004. At n=600 across 23 groups there
is not enough signal to fit weights. The gain comes from pool diversity, not from weighting.

**Rank-averaging fails** (-0.039 `spk_sim`, -0.052 `acc_sim`). The members' calibrations agree
well enough that discarding magnitudes loses more than it gains.

**Grouped CV is optimistic relative to dev** — agreement r=0.774 (`spk_sim`), r=0.652
(`acc_sim`), with CV consistently higher. This is the unseen-system effect, and it is why
selection uses `dev_srcc`, not `cv_srcc`.

---

## Layout and usage

```
extract_features.py   frozen embeddings for 4,160 unique wavs -> egs/features/*.npz
features.py           pair matrices, feature sets, train-only PCA
train_weak.py         5 learner families x GroupKFold -> egs/weak/preds/, resumable manifest
analyze.py            six metrics, residual correlations, ensemble comparison  (phase 1)
stack.py              combine deep + weak, write submission                    (phase 2)
```

```bash
# everything, on Slurm (~10 min GPU + ~6 h CPU)
sbatch ../jobs/weak/voicemos-track3-weak-phase1.sh

# or by hand
python extract_features.py --list
python extract_features.py --encoders sv:eres2netv2 ssl:WAVLM_LARGE --ssl-layers 4 8 -1
python train_weak.py --features "egs/features/*.npz" --targets spk_sim
python analyze.py --target spk_sim
python stack.py --target spk_sim --require-test --write-dev egs/submission --write-test egs/submission
```

Predictions are `egs/weak/preds/<encoder>__<featureset>__<learner>__<metric>__<split>.csv`
for `split` in `{oof, dev, test}`. `oof` is out-of-fold on train (2,800 rows) and is what a
phase-2 stacker would need if the deep models ever get out-of-fold predictions too.

### On phase 2

`stack.py` fits the combiner on **dev**, and reports its `GroupKFold`-by-system out-of-fold
score. UTMOS22 instead stacks on out-of-fold predictions from every level-0 model, which we
cannot do: the deep models were each trained on all of train, and generating honest OOF
predictions would mean K-fold retraining 32 models at 47 min to 5 h each — over 200
GPU-hours. The submitted weights are refitted on all 600 dev pairs; this is recorded in each
manifest rather than hidden.

Two caveats that belong in any write-up. Member **selection** (top-8 per pool by dev SRCC)
happens outside the CV loop, so the out-of-fold figures carry roughly 0.00-0.01 of selection
optimism, estimated from earlier split-half analysis. And `nnls`/`ridge` weights are fitted on
600 pairs across 23 groups, which is why the zero-parameter `mean` is the safer submission
despite `nnls` scoring 0.008 higher on `acc_sim`.

---

## References

### Papers

| | |
|---|---|
| **UTMOS** — Saeki et al., *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022*, Interspeech 2022. The stacking recipe this directory implements. | [arXiv:2204.02152](https://arxiv.org/abs/2204.02152) · [local PDF](../papers/UTMOS%3A%20UTokyo-SaruLab%20System%20for%20VoiceMOS%20Challenge%202022.pdf) · [code](../papers/UTMOS22) |
| **ECAPA-TDNN** — Desplanques et al., Interspeech 2020. | [arXiv:2005.07143](https://arxiv.org/abs/2005.07143) · [local PDF](../papers/ECAPA-TDNN%3A%20Emphasized%20Channel%20Attention%2C%20Propagation%20and%20Aggregation%20in%20TDNN%20Based%20Speaker%20Verification.pdf) |
| **ERes2NetV2** — Chen et al., Interspeech 2024. | [arXiv:2406.02167](https://arxiv.org/abs/2406.02167) · [local PDF](../papers/ERes2NetV2%3A%20Boosting%20Short-Duration%20Speaker%20Verification%20Performance%20with%20Computational%20Efficiency.pdf) |
| **CommonAccent** — Zuluaga-Gomez et al., Interspeech 2023. | [arXiv:2305.18283](https://arxiv.org/abs/2305.18283) |
| **WavLM** — Chen et al., IEEE JSTSP 2022. | [arXiv:2110.13900](https://arxiv.org/abs/2110.13900) |
| **wav2vec 2.0** — Baevski et al., NeurIPS 2020. | [arXiv:2006.11477](https://arxiv.org/abs/2006.11477) |
| **XLS-R** — Babu et al., 2021. Source of `WAV2VEC2_XLSR_300M`. | [arXiv:2111.09296](https://arxiv.org/abs/2111.09296) |
| **HuBERT** — Hsu et al., 2021. Not used yet; the natural fourth SSL bundle. | [arXiv:2106.07447](https://arxiv.org/abs/2106.07447) |
| **Tseng et al.** — *Utilizing Self-supervised Representations for MOS Prediction*, Interspeech 2021. Frozen wav2vec 2.0 features give UTT-LCC 0.734 vs 0.215 for mel-spectrograms — the prior evidence that frozen SSL features carry this signal. | [local PDF](../papers/Utilizing%20Self-supervised%20Representations%20for%20MOS%20Prediction.pdf) |
| **LightGBM** — Ke et al., NeurIPS 2017. Replaced here by sklearn's equivalent. | [code](https://github.com/microsoft/LightGBM) |
| **scikit-learn** — Pedregosa et al., JMLR 2011. | [code](https://github.com/scikit-learn/scikit-learn) |

### Implementations

- [sarulab-speech/UTMOS22](https://github.com/sarulab-speech/UTMOS22) — vendored at
  [../papers/UTMOS22](../papers/UTMOS22). The stage-1/2/3 design, the five learner families
  and mean-pooled SSL features all come from
  [`stacking/ensemble_multidomain_scripts/`](../papers/UTMOS22/stacking/ensemble_multidomain_scripts/).
- [pytorch/audio pipelines](https://docs.pytorch.org/audio/stable/pipelines.html) — all SSL
  bundles, replacing UTMOS22's fairseq checkpoints.
- [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) — ECAPA and
  CommonAccent checkpoints.
- [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) — ERes2NetV2, vendored at
  [../papers/3D-Speaker](../papers/3D-Speaker).

### Related directories

- [../unified/](../unified/) — the deep models this pool is combined with.
- [../utmos-approach/](../utmos-approach/) — UTMOS's *loss*, the other half of the same paper.
- [../papers/IDEAS.md](../papers/IDEAS.md) — the survey this experiment was selected from.
- [../jobs/weak/](../jobs/weak/) — the Slurm drivers: `voicemos-track3-weak-phase1.sh` runs
  everything above; `voicemos-track3-deep-test-inference.sh` generates the deep pool's test
  predictions, which `stack.py --require-test` needs in order to write a submission.
