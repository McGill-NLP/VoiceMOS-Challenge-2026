# Frozen-feature weak learners and stacking (UTMOS22's other half)

Classical regressors — ridge, SVR, random forest, gradient boosting — fitted on **frozen**
encoder embeddings, then pooled with the fine-tuned deep models from
[../unified/](../unified/).

This is the *stacking* half of UTMOS22 (Saeki et al., Interspeech 2022). The *loss* half of
the same paper is a separate experiment in [../utmos-approach/](../utmos-approach/); the two
are independent and can be read in either order.

Nothing here is fine-tuned. Every encoder runs once, under `torch.inference_mode()`, and the
embeddings are cached to disk. All the learning happens in scikit-learn on the CPU.

**Result: `spk_sim` dev UTT-SRCC 0.623 and `acc_sim` 0.603**, against 0.579 / 0.564 for the
deep pool alone — and 0.607 / 0.579 for the best previous system. These are the first ensemble
gains in the project whose 95% CIs exclude zero: **+0.043 [+0.020, +0.067]** and
**+0.039 [+0.012, +0.068]**. The shipped system is the deep top-8 plus the weak top-16,
combined by an unweighted mean with no fitted parameters.

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

Feature matrices are standardised on train; SVR and ridge are both scale-sensitive.

**PCA is off by default, and turning it on was a mistake worth recording.** The first sweep
reduced any embedding wider than 256 to 128 principal components, on the reasoning that 1024-d
SSL embeddings give 4,098 `full` features against 2,800 pairs. Both halves of that reasoning
were wrong. Ridge already handles p > n — its selected alpha rose to 10,000 on the raw
features, i.e. it regularised itself — and the tree learners improved too, contradicting the
claim that they would drown. Measured on `WAV2VEC2_XLSR_300M_l4`:

| run | PCA-128 | raw | delta |
|---|---|---|---|
| ridge, `spk_sim` | 0.516 | **0.572** | +0.056 |
| hgb, `spk_sim` | 0.510 | **0.548** | +0.038 |
| ridge, `acc_sim` | 0.452 | **0.505** | +0.053 |

Across the SSL sweep 23 of 24 configurations improved without PCA, by +0.05 to +0.29. PCA
maximises retained *variance*, and the dominant variance directions in an SSL embedding are
speaker, channel and recording characteristics — not the fine distinctions this judgement
needs. Use `--pca-threshold 99999` to disable it, which is what the shipped results use.

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

**Targets** are the per-pair mean over raters: 13,687 rating rows collapse to 2,800 pairs.
The feature vector is identical across a pair's rows, so for a squared-error learner this is
exactly equivalent to weighting pairs by rater count, and the counts are nearly uniform (2,488
pairs with 5 ratings, 311 with 4, 1 with 3). `--per-rating` fits the individual ratings
instead; it was tested and is slightly WORSE (see Training-set variants below), because the CV
objective over 13,687 rows selects different, poorer regularisation.

---

## Results

All on `dev_with_labels.csv`, 600 pairs.

### Best single weak models (no PCA, fitted on train, scored on held-out dev)

| target | encoder | features | learner | uSRCC |
|---|---|---|---|---|
| `spk_sim` | `WAVLM_LARGE_l4` | full | ridge | **0.602** |
| `spk_sim` | `WAVLM_LARGE_l4` | compact | ridge | 0.583 |
| `spk_sim` | `WAV2VEC2_XLSR_300M_l4` | compact | ridge | 0.573 |
| `acc_sim` | `WAVLM_LARGE_l4` | compact | ridge | **0.542** |
| `acc_sim` | `WAVLM_LARGE_l4` | full | ridge | 0.541 |
| `acc_sim` | `WAVLM_BASE_PLUS_l4` | full | ridge | 0.531 |

For reference, the best fine-tuned deep models are 0.569 (`spk_sim`) and 0.552 (`acc_sim`).

### Pools

All rows are the unweighted mean of their members, scored on `dev_with_labels.csv`. The
deep members are identical throughout (fitted on train only); only the weak half changes.

**Read the `[IN-SAMPLE]` rows as diagnostics, not results.** When the weak learners are
refitted on train+dev, dev becomes 600 of the 3,400 fitting pairs, so those numbers are partly
memorisation. They are recorded because they were asked for and because the size of the
inflation is itself informative — not because they estimate anything.

| pool | uMSE | uLCC | uSRCC | sMSE | sLCC | sSRCC | vs deep-only, 95% CI |
|---|---|---|---|---|---|---|---|
| **`spk_sim`** | | | | | | | |
| deep top-8 (train only) | 0.349 | 0.621 | 0.579 | **0.049** | 0.938 | 0.907 | — |
| weak top-8, train only *(held out)* | 0.354 | 0.639 | 0.610 | 0.066 | **0.954** | **0.941** | +0.030 [-0.012, +0.072] |
| weak top-16, train only *(held out)* | 0.339 | 0.653 | 0.622 | 0.055 | 0.951 | 0.917 | +0.043 [+0.009, +0.080] |
| deep + weak top-8, train only *(held out)* | 0.337 | 0.652 | 0.620 | 0.053 | 0.952 | 0.931 | +0.040 [+0.020, +0.061] |
| **deep + weak top-16, train only** *(held out)* | **0.333** | **0.656** | **0.623** | 0.051 | 0.952 | 0.929 | **+0.043 [+0.020, +0.067]** |
| weak top-8, train+dev `[IN-SAMPLE]` | 0.229 | 0.792 | 0.766 | 0.028 | 0.968 | 0.960 | — |
| deep + weak top-8, train+dev `[IN-SAMPLE]` | 0.274 | 0.733 | 0.701 | 0.036 | 0.960 | 0.947 | — |
| deep + weak top-16, train+dev `[IN-SAMPLE]` | 0.200 | 0.832 | 0.812 | 0.026 | 0.973 | 0.947 | — |
| **`acc_sim`** | | | | | | | |
| deep top-8 (train only) | 0.316 | 0.601 | 0.564 | 0.039 | 0.924 | 0.933 | — |
| weak top-8, train only *(held out)* | 0.346 | 0.581 | 0.560 | 0.057 | 0.942 | **0.957** | -0.004 [-0.052, +0.043] |
| weak top-16, train only *(held out)* | 0.317 | 0.623 | 0.597 | 0.040 | 0.936 | 0.944 | +0.032 [-0.008, +0.072] |
| deep + weak top-8, train only *(held out)* | 0.314 | 0.624 | 0.595 | 0.040 | **0.942** | 0.958 | +0.031 [+0.004, +0.057] |
| **deep + weak top-16, train only** *(held out)* | **0.307** | **0.636** | **0.603** | **0.038** | 0.937 | 0.943 | **+0.039 [+0.012, +0.068]** |
| weak top-8, train+dev `[IN-SAMPLE]` | 0.245 | 0.731 | 0.694 | 0.021 | 0.956 | 0.959 | — |
| deep + weak top-8, train+dev `[IN-SAMPLE]` | 0.267 | 0.699 | 0.660 | 0.028 | 0.948 | 0.959 | — |
| deep + weak top-16, train+dev `[IN-SAMPLE]` | 0.194 | 0.818 | 0.786 | 0.020 | 0.965 | 0.963 | — |

**k=16 is the shipped configuration** and is what `make_submission.py` produces by default.

Weak-only is *not* a substitute for the combination. At k=8 it reaches 0.610 on `spk_sim` but
with a CI that straddles zero, and on `acc_sim` it is flat (-0.004). Only the combination has
an interval excluding zero on both targets, at either k.

### Pool size: why 16 and not everything

Sweeping the number of weak members, deep top-8 held fixed, held-out dev:

| k | worst member | resid corr | weak-only uSRCC | deep+weak uSRCC | vs deep-only |
|---|---|---|---|---|---|
| **`spk_sim`** | | | | | |
| 4 | 0.572 | 0.967 | 0.601 | 0.613 | +0.034 [+0.020, +0.049] |
| 8 | 0.534 | 0.925 | 0.610 | 0.620 | +0.040 [+0.019, +0.061] |
| **16** | 0.498 | 0.871 | **0.622** | **0.623** | **+0.044 [+0.021, +0.068]** |
| 24 | 0.470 | 0.856 | 0.611 | 0.616 | +0.037 [+0.010, +0.065] |
| 32 | 0.458 | 0.838 | 0.601 | 0.608 | +0.029 [-0.002, +0.059] |
| 40 | 0.421 | 0.825 | 0.595 | 0.602 | +0.022 [-0.010, +0.057] |
| 48 | 0.382 | 0.811 | 0.591 | 0.598 | +0.019 [-0.014, +0.054] |
| 58 (all) | 0.311 | 0.788 | 0.579 | 0.588 | +0.009 [-0.025, +0.044] |
| **`acc_sim`** | | | | | |
| 4 | 0.528 | 0.954 | 0.563 | 0.593 | +0.029 [+0.010, +0.046] |
| 8 | 0.516 | 0.955 | 0.560 | 0.595 | +0.031 [+0.004, +0.058] |
| **16** | 0.489 | 0.877 | **0.597** | **0.603** | **+0.039 [+0.013, +0.067]** |
| 24 | 0.473 | 0.866 | 0.594 | 0.597 | +0.033 [+0.007, +0.059] |
| 32 | 0.446 | 0.862 | 0.592 | 0.594 | +0.030 [+0.006, +0.055] |
| 40 | 0.428 | 0.841 | 0.585 | 0.590 | +0.026 [-0.003, +0.055] |
| 48 | 0.375 | 0.831 | 0.581 | 0.586 | +0.021 [-0.008, +0.049] |
| 58 (all) | 0.250 | 0.806 | 0.569 | 0.576 | +0.011 [-0.018, +0.040] |

Residual correlation falls monotonically with k, so by that measure the full pool is the most
diverse available — and it performs worst, barely above deep-only. Meanwhile k=4 has the best
members and the least diversity, and also underperforms. **Neither diversity nor member
quality is the objective; the peak is where both are adequate.** Past k=16 an unweighted mean
gives a 0.31-scoring model the same vote as a 0.60-scoring one, and the tail of `linsvr` and
`rf` models actively destroys the ranking.

Note the k=8 -> 16 gap is small (+0.003 `spk_sim`, +0.008 `acc_sim`), well inside the ±0.05
measurement floor. "16 beats 8" is a soft claim; "16 beats 58" is a firm one. This is also
where a weighted combiner should have earned its keep by downweighting the tail instead of
truncating it — `nnls` and `ridge` landed within 0.004 of the unweighted mean, so truncation
is doing the job fitted weights could not.

### Weak top-16 membership

Ranks 9-16 all score *below* the top 8 individually, and they are what makes k=16 work:
they introduce `ksvr`, `hgb` and the speaker-ID encoders into a top-8 that is almost entirely
ridge-on-SSL.

| rank | `spk_sim` | uSRCC | `acc_sim` | uSRCC |
|---|---|---|---|---|
| 1 | `WAVLM_LARGE_l4__full__ridge` | 0.602 | `WAVLM_LARGE_l4__compact__ridge` | 0.542 |
| 2 | `WAVLM_LARGE_l4__compact__ridge` | 0.583 | `WAVLM_LARGE_l4__full__ridge` | 0.541 |
| 3 | `WAV2VEC2_XLSR_300M_l4__compact__ridge` | 0.573 | `WAVLM_BASE_PLUS_l4__full__ridge` | 0.531 |
| 4 | `WAV2VEC2_XLSR_300M_l4__full__ridge` | 0.572 | `WAVLM_LARGE_l8__compact__ridge` | 0.528 |
| 5 | `WAVLM_LARGE_l8__full__ridge` | 0.545 | `WAV2VEC2_XLSR_300M_l8__compact__ridge` | 0.524 |
| 6 | `WAV2VEC2_XLSR_300M_l8__full__ridge` | 0.542 | `WAV2VEC2_XLSR_300M_l8__full__ridge` | 0.523 |
| 7 | `WAVLM_BASE_PLUS_l4__full__ridge` | 0.535 | `WAVLM_LARGE_l8__full__ridge` | 0.521 |
| 8 | `eres2netv2__full__ksvr` | 0.534 | `WAV2VEC2_XLSR_300M_l4__compact__ridge` | 0.516 |
| 9 | `WAVLM_LARGE_l8__compact__ridge` | 0.533 | `WAVLM_BASE_PLUS_l4__compact__ridge` | 0.509 |
| 10 | `eres2netv2__full__ridge` | 0.527 | `WAV2VEC2_XLSR_300M_l4__full__ridge` | 0.505 |
| 11 | `eres2netv2__full__hgb` | 0.515 | `ecapa-voxceleb__full__ridge` | 0.501 |
| 12 | `eres2netv2-w24s4ep4__full__ksvr` | 0.511 | `commonaccent-ecapa__full__hgb` | 0.499 |
| 13 | `eres2netv2-w24s4ep4__full__hgb` | 0.503 | `ecapa-voxceleb__full__ksvr` | 0.499 |
| 14 | `ecapa-voxceleb__full__hgb` | 0.500 | `WAVLM_LARGE_l24__full__ridge` | 0.495 |
| 15 | `WAVLM_BASE_PLUS_l4__compact__ridge` | 0.499 | `ecapa-voxceleb__compact__hgb` | 0.491 |
| 16 | `eres2netv2__compact__ksvr` | 0.498 | `ecapa-voxceleb__full__hgb` | 0.489 |

All 16 of the `acc_sim` top-8 and 7 of the `spk_sim` top-8 are SSL+ridge; no speaker-ID
encoder reaches the `acc_sim` top 8 at all. These lists are frozen in
[make_submission.py](make_submission.py) and must not be re-derived from any run where dev
was in training.

### Training-set variants: per-rating targets and outlier removal

Two attempts to improve the weak half by changing the *training data* rather than the models.
Members, learners and hyperparameter grids are unchanged; only the fitting set differs, and
dev stays held out in both, so all rows are directly comparable.

- **per-rating** — fit the 13,687 individual listener ratings instead of the 2,800 per-pair
  means ([../jobs/weak/voicemos-track3-weak-per-rating.sh](../jobs/weak/voicemos-track3-weak-per-rating.sh)).
  `ksvr` is absent (a 13,687^2 kernel is infeasible) and `linsvr`/`rf` were skipped for cost,
  so this pool has 13/16 and 15/16 members.
- **no-outliers** — `sets/train-without-outliers.csv`
  ([../jobs/weak/voicemos-track3-weak-outliers.sh](../jobs/weak/voicemos-track3-weak-outliers.sh)).
  Despite the name it removes two whole **listeners**, not scattered ratings: 25 -> 23
  listeners, 1,097 of 13,687 rows (8.0%), and **zero pairs** — all 2,800 survive. Each pair
  averages ~4.5 ratings instead of ~4.9 (spk_sim mean 4.048 -> 4.019, sd 1.190 -> 1.205).

**Deep top-8 + weak top-16**, held-out dev:

| pool | n | uMSE | uLCC | uSRCC | sMSE | sLCC | sSRCC | residCorr | vs ORIGINAL |
|---|---|---|---|---|---|---|---|---|---|
| **`spk_sim`** | | | | | | | | | |
| deep top-8 only | 8 | 0.349 | 0.621 | 0.579 | 0.049 | 0.938 | 0.907 | 0.957 | -0.043 [-0.068, -0.019] |
| **ORIGINAL** | 24 | 0.333 | 0.656 | **0.623** | 0.051 | 0.952 | 0.929 | 0.877 | — |
| per-rating | 21 | 0.335 | 0.654 | 0.621 | 0.051 | **0.955** | **0.932** | 0.853 | -0.002 [-0.012, +0.007] |
| no-outliers | 24 | **0.329** | 0.656 | 0.621 | **0.046** | 0.953 | 0.930 | 0.871 | -0.001 [-0.005, +0.002] |
| orig + no-outliers | 40 | 0.332 | **0.657** | **0.625** | 0.049 | 0.953 | 0.926 | 0.871 | +0.002 [-0.004, +0.008] |
| all variants pooled | 53 | 0.333 | **0.657** | **0.625** | 0.050 | **0.955** | 0.921 | 0.860 | +0.003 [-0.005, +0.011] |
| **`acc_sim`** | | | | | | | | | |
| deep top-8 only | 8 | 0.316 | 0.601 | 0.564 | 0.039 | 0.924 | 0.933 | 0.927 | -0.038 [-0.066, -0.010] |
| **ORIGINAL** | 24 | 0.307 | **0.636** | **0.603** | 0.038 | 0.937 | 0.943 | 0.872 | — |
| per-rating | 23 | 0.310 | 0.626 | 0.591 | 0.037 | **0.941** | 0.935 | 0.836 | **-0.012 [-0.023, -0.001]** |
| no-outliers | 24 | **0.303** | **0.636** | 0.600 | **0.033** | 0.939 | 0.943 | 0.863 | -0.003 [-0.007, +0.001] |
| orig + no-outliers | 40 | 0.307 | 0.634 | **0.603** | 0.035 | 0.938 | 0.943 | 0.868 | +0.000 [-0.006, +0.006] |
| all variants pooled | 55 | 0.310 | 0.629 | 0.599 | 0.036 | 0.940 | **0.946** | 0.856 | -0.004 [-0.013, +0.005] |

**Weak only**, same variants with the deep members removed:

| pool | n | uMSE | uLCC | uSRCC | sMSE | sLCC | sSRCC | residCorr | vs ORIGINAL |
|---|---|---|---|---|---|---|---|---|---|
| **`spk_sim`** | | | | | | | | | |
| **ORIGINAL** | 16 | 0.339 | **0.653** | **0.622** | 0.055 | 0.951 | 0.917 | 0.871 | — |
| per-rating | 13 | 0.346 | 0.646 | 0.612 | 0.056 | **0.958** | 0.916 | 0.832 | -0.010 [-0.025, +0.005] |
| no-outliers | 16 | **0.333** | 0.652 | 0.620 | **0.047** | 0.952 | 0.915 | 0.863 | -0.002 [-0.008, +0.003] |
| orig + no-outliers | 32 | 0.336 | **0.653** | 0.621 | 0.051 | 0.951 | 0.915 | 0.871 | -0.001 [-0.004, +0.002] |
| orig + per-rating | 29 | 0.340 | **0.653** | **0.622** | 0.055 | 0.955 | **0.920** | 0.854 | -0.000 [-0.007, +0.007] |
| all variants pooled | 45 | 0.337 | **0.653** | **0.623** | 0.052 | 0.954 | 0.917 | 0.859 | +0.001 [-0.005, +0.006] |
| **`acc_sim`** | | | | | | | | | |
| **ORIGINAL** | 16 | 0.317 | 0.623 | **0.597** | 0.040 | 0.936 | 0.944 | 0.877 | — |
| per-rating | 15 | 0.325 | 0.608 | 0.579 | 0.043 | **0.943** | 0.927 | 0.824 | **-0.018 [-0.035, -0.003]** |
| no-outliers | 16 | **0.309** | **0.624** | 0.594 | **0.032** | 0.939 | 0.952 | 0.863 | -0.003 [-0.009, +0.003] |
| orig + no-outliers | 32 | 0.313 | **0.624** | 0.595 | 0.036 | 0.937 | **0.954** | 0.874 | -0.001 [-0.005, +0.002] |
| orig + per-rating | 31 | 0.320 | 0.619 | 0.591 | 0.041 | 0.940 | 0.944 | 0.852 | -0.005 [-0.013, +0.003] |
| all variants pooled | 47 | 0.315 | 0.621 | 0.594 | 0.038 | 0.940 | 0.946 | 0.858 | -0.002 [-0.008, +0.003] |

Best single member per variant: ORIGINAL 0.602 / 0.542, no-outliers 0.600 / 0.537,
per-rating 0.565 / 0.512.

**Neither variant beats the original, and pooling them recovers parity at best.** The only
interval excluding zero is per-rating on `acc_sim`, in the wrong direction.

**Per-rating is a small loss, and not for the reason predicted.** The algebra says duplicated
rows are equivalent to weighting pairs by rater count, so ridge should have been a control
landing within +/-0.01. It moved further, and consistently downward (0.602 -> 0.565 on the
best member). The loss function is not what changed — the CV objective is now computed over
13,687 rows instead of 2,800, which shifts which alpha wins, and it selects worse
regularisation. Part of the pool gap is also the three missing `ksvr` members.

**Outlier removal is a genuine no-op for ranking and a real gain for error.** SRCC moves
-0.001 to -0.003 with the tightest intervals in the table, while MSE improves everywhere:
weak-only `acc_sim` uMSE 0.317 -> **0.309** and sMSE 0.040 -> **0.032**, a 20% reduction.
Dropping two noisy raters tightens the targets' absolute scale without reordering them —
exactly what the mechanism predicts. If the final metric combination weights MSE, the
no-outliers weak half is defensible at a cost of ~0.003 SRCC.

**Refitting the same estimators on perturbed targets does not buy diversity.** Cross-variant
residual correlation is 0.874 (original vs no-outliers) and 0.853 (vs per-rating) — about the
same as *within* the original pool (0.877). That is why pooling 45-55 members gains nothing:
the variants are not making different mistakes, only slightly worse ones.

**Effect of the PCA bug** (see Features): with PCA-128, the same held-out pools scored
`spk_sim` 0.609 and `acc_sim` 0.577 instead of 0.620 and 0.595.

**Size of the in-sample inflation.** Per member it is +0.10 to +0.19 for the ridge models, and
**+0.388** for `eres2netv2__full__ksvr` (0.534 -> 0.923): an RBF-SVR interpolates its own
training points, so on rows it was fitted on it looks near-perfect. That single number is the
cleanest demonstration in this project of why the protocol matters. Note also that the
`deep + weak` in-sample rows are *lower* than `weak`-only in-sample, because the eight deep
members never saw dev and pull the average back toward honesty.

The best available estimate for the train+dev system remains the held-out **0.620 / 0.595**
plus an unmeasurable gain from 21% more data. Test predictions from the two variants correlate
at 0.998 (`spk_sim`) and 0.997 (`acc_sim`), so whatever that gain is, it is small.

### Decorrelation — the thing the experiment was actually testing

Mean pairwise correlation of member residuals:

| | `spk_sim` | `acc_sim` |
|---|---|---|
| within deep top-8 | 0.957 | 0.927 |
| within weak top-8, PCA-128 | 0.851 | 0.867 |
| within weak top-8, no PCA | 0.925 | 0.955 |
| **within weak top-16, no PCA** | **0.871** | **0.877** |
| deep vs weak top-8, cross-pool (PCA-128) | 0.866 | 0.859 |
| deep vs weak top-8, cross-pool (no PCA) | 0.871 | 0.865 |
| **deep vs weak top-16, cross-pool (no PCA)** | **0.864** | **0.855** |

The cross-pool figure is the one that matters, and under both settings it sits at or below the
0.878 that the heterogeneous deep pool needed to earn +0.062 — well below the 0.957 the deep
pool manages internally. Changing the function class decorrelates the errors in a way that
swapping one speaker-ID encoder for another did not. The mechanism worked.

---

## Findings

**Early SSL layers beat late ones, but less than PCA made it look.** Layer 4 wins for every
bundle, and UTMOS22 used only the last layer. Under PCA-128 the gap looked enormous —
`XLSR_300M` 0.516 at layer 4 vs 0.438 at layer 24 (`spk_sim`) — but removing PCA lifted the
deep layers by +0.244 and +0.292, so much of that gap was PCA damage rather than layer
quality. The ordering survives; the magnitude does not.

**Frozen SSL features beat fine-tuned encoders.** The single best model anywhere in this
project is `WAVLM_LARGE_l4__full__ridge` at **0.602** on `spk_sim` — a ridge regression on
frozen WavLM-Large layer-4 embeddings, fitted in about ten seconds. It beats every fine-tuned
deep model (best 0.569) and nearly matches the entire deep top-8 ensemble (0.579). On 2,800
pairs, gradient fine-tuning is contributing much less than assumed.

**A multilingual SSL model beats the purpose-built accent encoder.** On `acc_sim`,
`WAV2VEC2_XLSR_300M` and `WAVLM_LARGE` layer-4 models reach 0.516-0.542, ahead of
`commonaccent-ecapa` (0.499). XLS-R treats accent variation as signal; an accent *classifier*
compresses it to 16 logits.

**`full` vs `compact` is a wash once PCA is removed.** Under PCA `full` won everywhere (0.534
vs 0.498 on `spk_sim`). On raw features they trade places — `XLSR_300M_l4` scores 0.573
compact vs 0.572 full — so the raw embedding blocks are not clearly earning their place.

**Better members can mean a worse pool, and widening the pool fixes it.** Removing PCA raised
individual scores but *raised* internal residual correlation within the weak top-8, from 0.851
to 0.925 (`spk_sim`) and 0.867 to 0.955 (`acc_sim`), because the no-PCA top-8 is almost
entirely ridge on similar SSL layers. Going to k=16 pulls it back to 0.871 / 0.877 by adding
`ksvr`, `hgb` and the speaker-ID encoders — members that score *lower* individually. So the
no-PCA gain came from member *quality* and the k=16 gain from *diversity*; they are separate
mechanisms and they compose.

**More members is not monotonically better.** Using all 58 candidates scores 0.588 / 0.576,
barely above deep-only and well below k=16, despite having the lowest residual correlation of
any pool tried (0.788 / 0.806). An unweighted mean gives a 0.31-scoring model the same vote as
a 0.60-scoring one. See the pool-size sweep above.

**Learned stacking weights buy nothing over a plain average.** On `spk_sim` the unweighted
mean *won*, with `alpha`, `nnls` and `ridge` all within 0.004. At n=600 across 23 groups there
is not enough signal to fit weights. The gain comes from pool diversity, not from weighting.

**Rank-averaging fails** (-0.039 `spk_sim`, -0.052 `acc_sim`). The members' calibrations agree
well enough that discarding magnitudes loses more than it gains.

**Grouped CV is optimistic relative to dev** — agreement r=0.774 (`spk_sim`), r=0.652
(`acc_sim`), with CV consistently higher. This is the unseen-system effect, and it is why
selection uses `dev_srcc`, not `cv_srcc`.

**Changing the training data did not help; changing the model class did.** Two interventions
on the fitting set — individual listener ratings instead of pair means, and dropping two
unreliable listeners — both came back neutral-to-negative on SRCC, and pooling their outputs
with the originals gained nothing because the variants correlate with the original members
about as strongly as those members correlate with each other. Set against the +0.043 / +0.039
this directory earned by swapping fine-tuned encoders for frozen features and classical
regressors, the pattern is consistent: at 2,800 pairs the leverage is in the function class,
not in the last 8% of the labels. The one exception is absolute error — outlier removal cuts
`acc_sim` system-level MSE by 20% while leaving the ranking untouched.

---

## Layout and usage

```
extract_features.py   frozen embeddings for 4,160 unique wavs -> egs/features/*.npz
features.py           pair matrices, feature sets, optional train-only PCA
train_weak.py         5 learner families x GroupKFold -> <outdir>/preds/, resumable manifest
analyze.py            six metrics, residual correlations, ensemble comparison  (phase 1)
stack.py              combine deep + weak, SELECTS members by dev SRCC          (phase 2)
make_submission.py    combine from an EXPLICIT frozen member list, selects nothing
```

Output trees, none of which overwrite another:

| tree | fitted on | notes |
|---|---|---|
| `egs/weak/` | train, pair means | the original 260-model sweep, with PCA-128 on SSL |
| `egs/weak_nopca/` | train, pair means | no PCA; **the shipped held-out results** (58 candidates per target) |
| `egs/weak_traindev/` | train+dev, pair means | 84 refits; every record carries `dev_in_train: true` |
| `egs/weak_perrating/` | train, 13,687 rating rows | the `--per-rating` experiment (slightly worse) |
| `egs/weak_outliers/` | train minus 2 listeners | `train-without-outliers.csv`; SRCC-neutral, better MSE |
| `egs/submission_final/` | — | four packaged variants, named `deep<D>-weak<W>_<fitting set>` |

```bash
# the full held-out sweep, on Slurm (~10 min GPU + ~6 h CPU)
sbatch ../jobs/weak/voicemos-track3-weak-phase1.sh

# by hand
python extract_features.py --list
python extract_features.py --encoders sv:eres2netv2 ssl:WAVLM_LARGE --ssl-layers 4 8 -1
python train_weak.py --features "egs/features/*.npz" --targets spk_sim --pca-threshold 99999
python analyze.py --target spk_sim
python stack.py --target spk_sim --require-test --write-dev egs/submission --write-test egs/submission

# refit on train+dev (allowed by the challenge) and build the submission from the FROZEN list
python train_weak.py --train-csv ../baseline/data/vmc2026_track3_train_phase_distro_v3_syn/sets/train_plus_dev.csv \
    --features "egs/features/*.npz" --pca-threshold 99999 --outdir egs/weak_traindev
# k defaults to the measured optimum: --deep-k 8 --weak-k 16
python make_submission.py --target spk_sim --weak-dir egs/weak_traindev/preds \
    --out egs/submission_final/deep8-weak16_train_plus_dev

# training-set variants (both CPU-only; ~1 h and ~25 min)
sbatch ../jobs/weak/voicemos-track3-weak-per-rating.sh
sbatch ../jobs/weak/voicemos-track3-weak-outliers.sh
```

Submission folders are named for their composition, `deep<D>-weak<W>_<fitting set>`, so
`deep8-weak16_train_only/` is the deep top-8 PLUS the weak top-16 (24 members) with the weak
half fitted on train. Every submission in this directory contains deep members; there is no
weak-only variant on disk. `make_submission.py --deep-k 0` would build one.

Predictions are `<outdir>/preds/<encoder>__<featureset>__<learner>__<metric>__<split>.csv`
for `split` in `{oof, dev, test}`. `oof` is out-of-fold on the fitting set and is what a
phase-2 stacker would need if the deep models ever get out-of-fold predictions too.

### Training on train+dev

`sets/train_plus_dev.csv` (3,400 pairs, 23 systems) concatenates `train.csv` and
`dev_with_labels.csv`; dev rows carry an empty `listener_id`, which the deep pipeline drops
anyway when `keep_listeners=False`. No feature recomputation or wav copying is needed — the
cache already covers all 4,160 wavs including dev's, and every dev wav resolves under the
train distribution's root.

`train_weak.py` detects from the pair keys, not from a flag, whether dev is inside the fitting
set, and then stamps `dev_in_train: true` into every manifest record and tags each log line
`[IN-SAMPLE]`. This is not cosmetic: with dev in training there is **no held-out set left**, so
member selection cannot be redone and must stay frozen at what the held-out runs chose. That
is why `make_submission.py` exists — `stack.py` would re-select on the in-sample scores and
pick whichever model memorised hardest.

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
