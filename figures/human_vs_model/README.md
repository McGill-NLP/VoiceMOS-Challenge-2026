# Human ratings versus model predictions

From the repository root, after installing `figures/requirements.txt`:

```bash
python3 figures/human_vs_model/plot_human_vs_model.py
```

The default inputs are the 600 official test pairs and the submitted weak-16
predictions from `deep0-weak16_train_plus_dev`. These models were fitted on
train+dev; the plotted test pairs are held out. Human scores are the released
per-pair mean opinion scores, not individual listener ratings. This is a
descriptive analysis of the fixed submission, not model selection.

Panel (a) plots human speaker MOS against human accent MOS. Panel (b) plots the
model's speaker predictions against its accent predictions for exactly the same
pairs. Both use identical 1–5 axes and one point per pair, with the same point
size and transparency. Overlapping points appear darker; exact duplicates can
overlap completely. No binning, jitter, or smoothing is applied. The dashed line
marks equal speaker and accent scores. Each pair has equal weight.

Statistics are omitted from the plots so they can be added in the paper's LaTeX
text or caption. For reference, the same **600 held-out test pairs** give:

| Statistic | Human MOS | Weak-16 predictions |
|---|---:|---:|
| Speaker–accent SRCC | 0.790469 | 0.896126 |
| Mean absolute speaker–accent gap (MOS points) | 0.363500 | 0.135843 |
| Number of pairs (n) | 600 | 600 |

Speaker–accent SRCC (previously labeled "cross-target SRCC") is the correlation
between the two attributes within each panel, not prediction accuracy against
human scores. The gap is the mean of `abs(accent - speaker)` across those pairs.

The model outputs are more strongly associated and have smaller differences
between dimensions. This does not by itself establish that the model cannot
distinguish the attributes: its scores also have less variation across pairs,
and averaging listener ratings and model predictions can suppress variation.
The figure does not measure prediction accuracy or explain why accent prediction
has lower accuracy. There are no confidence intervals or significance claims.

Outputs:

- `human_vs_model.pdf`, `.png` (300 dpi), and `.svg`.
- `human_vs_model.csv`: aligned human and model scores for all pairs.
- `summary.json`: exact statistics, per-target prediction accuracy, and input paths.

The script joins inputs on both waveform paths and rejects duplicate, missing,
or extra pairs and invalid scores. It accepts `--labels`, `--speaker-predictions`,
`--accent-predictions`, `--model-label`, and `--output-dir`. If changing the model
inputs, update `--model-label` accordingly. Default paths are relative to the
script, so it can run from any working directory.

Suggested caption:

> Speaker and accent similarity for the same 600 test pairs: (a) human mean
> opinion scores and (b) predictions of the submitted weak-16 ensembles fitted
> on train+dev. Each point represents one pair, with transparency revealing
> overlap; dashed lines indicate equal scores.
