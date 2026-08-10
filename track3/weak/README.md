# Frozen-feature weak learners and stacking (UTMOS22's other half)

Classical regressors — ridge, SVR, random forest, gradient boosting — fitted on **frozen**
encoder embeddings, then pooled with the fine-tuned deep models from
[../unified/](../unified/).

This is the *stacking* half of UTMOS22 (Saeki et al., Interspeech 2022). The *loss* half of
the same paper is a separate experiment in [../utmos-approach/](../utmos-approach/); the two
are independent and can be read in either order.

Nothing here is fine-tuned. Every encoder runs once, under `torch.inference_mode()`, and the
embeddings are cached to disk. All the learning happens in scikit-learn on the CPU.

**Result: `spk_sim` dev UTT-SRCC 0.620 and `acc_sim` 0.595**, against 0.579 / 0.564 for the
deep pool alone — and 0.607 / 0.579 for the best previous system. These are the first ensemble
gains in the project whose 95% CIs exclude zero: **+0.040 [+0.020, +0.061]** and
**+0.031 [+0.005, +0.058]**.

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

**Targets** are the per-pair mean over raters (train has ~5 rating rows per pair). Duplicating
identical feature rows would only reweight pairs by rater count.

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

| pool | uMSE | uLCC | uSRCC | sMSE | sLCC | sSRCC |
|---|---|---|---|---|---|---|
| **`spk_sim`** | | | | | | |
| deep top-8 (train only) | 0.349 | 0.621 | 0.579 | **0.049** | 0.938 | 0.907 |
| weak top-8, train only *(held out)* | 0.354 | 0.639 | 0.610 | 0.066 | **0.954** | **0.941** |
| **deep + weak, train only** *(held out)* | 0.337 | **0.652** | **0.620** | 0.053 | 0.952 | 0.931 |
| weak top-8, train+dev `[IN-SAMPLE]` | 0.229 | 0.792 | 0.766 | 0.028 | 0.968 | 0.960 |
| deep + weak, train+dev `[IN-SAMPLE]` | 0.274 | 0.733 | 0.701 | 0.036 | 0.960 | 0.947 |
| **`acc_sim`** | | | | | | |
| deep top-8 (train only) | 0.316 | 0.601 | 0.564 | **0.039** | 0.924 | 0.933 |
| weak top-8, train only *(held out)* | 0.346 | 0.581 | 0.560 | 0.057 | 0.942 | **0.957** |
| **deep + weak, train only** *(held out)* | 0.314 | **0.624** | **0.595** | 0.040 | **0.942** | 0.958 |
| weak top-8, train+dev `[IN-SAMPLE]` | 0.245 | 0.731 | 0.694 | 0.021 | 0.956 | 0.959 |
| deep + weak, train+dev `[IN-SAMPLE]` | 0.267 | 0.699 | 0.660 | 0.028 | 0.948 | 0.959 |

Gain over deep-only, bootstrap 95% CI, held-out rows only: `spk_sim`
**+0.040 [+0.020, +0.061]**, `acc_sim` **+0.031 [+0.005, +0.058]**. Both exclude zero.

Weak-only is *not* a substitute for the combination: on `spk_sim` it reaches 0.610 but at
+0.031 [-0.010, +0.074] against deep-only, and on `acc_sim` it is flat (-0.005). The
combination is what is reliably better on both targets.

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
| deep vs weak, cross-pool (PCA-128) | 0.866 | 0.859 |
| deep vs weak, cross-pool (no PCA) | 0.871 | 0.865 |

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

**Better members can mean a worse pool.** Removing PCA raised individual scores but *raised*
internal residual correlation within the weak top-8, from 0.851 to 0.925 (`spk_sim`) and 0.867
to 0.955 (`acc_sim`), because the no-PCA top-8 is dominated by ridge on similar SSL layers.
Cross-pool correlation against the deep models barely moved (0.866 -> 0.871). So the no-PCA
gain came from member *quality*, and the PCA pool's contribution was *diversity* — different
mechanisms, which is why the combined pool wins either way.

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
features.py           pair matrices, feature sets, optional train-only PCA
train_weak.py         5 learner families x GroupKFold -> <outdir>/preds/, resumable manifest
analyze.py            six metrics, residual correlations, ensemble comparison  (phase 1)
stack.py              combine deep + weak, SELECTS members by dev SRCC          (phase 2)
make_submission.py    combine from an EXPLICIT frozen member list, selects nothing
```

Output trees, none of which overwrite another:

| tree | fitted on | notes |
|---|---|---|
| `egs/weak/` | train | the original 260-model sweep, with PCA-128 on SSL |
| `egs/weak_nopca/` | train | SSL ridge without PCA; **the shipped held-out results** |
| `egs/weak_traindev/` | train+dev | 84 refits; every manifest record carries `dev_in_train: true` |

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
python make_submission.py --target spk_sim --weak-dir egs/weak_traindev/preds --out egs/submission_final/train_plus_dev
```

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
