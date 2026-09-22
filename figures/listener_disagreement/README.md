# Do listeners agree less about accent similarity?

From the repository root, after installing `figures/requirements.txt`:

```bash
python3 figures/listener_disagreement/plot_listener_disagreement.py
```

The script reads the official training CSV. Override the input with `--csv PATH`
or the output folder with `--output-dir PATH`.

For every unique `(wav_a_path, wav_b_path)` pair, compute the sample standard
deviation (`ddof=1`) of the individual speaker ratings and accent ratings. Both
dimensions use the same listeners. No listeners or pairs are removed, and each
pair contributes equally to the aggregate statistics.

The figure contains:

- **Panel (a):** empirical cumulative distributions of within-pair SD. A curve
  further to the right indicates greater rating dispersion. No smoothing is used.
- **Panel (b):** the distribution of accent SD minus speaker SD for the same pair.
  Positive values indicate greater disagreement about accent. The solid line is
  zero; the dashed orange line is the mean difference. The central histogram bin
  includes small nonzero differences as well as exact ties; the annotated tie
  percentage is calculated separately with a numerical tolerance of `1e-12`.

Outputs are `listener_disagreement.pdf`, `.png` (300 dpi), and `.svg`, plus
`listener_disagreement.csv` (one row per pair) and `summary.json` (aggregate
statistics and input path).

The default data has 13,687 ratings over 2,800 pairs: 2,488 pairs have five
listeners, 311 have four, and one has three. Mean within-pair SD is 0.847 for
speaker similarity and 0.919 for accent similarity (mean paired difference
0.073). Accent SD is higher on 41.25% of pairs, equal on 29.96%, and lower on 28.79%.

These are descriptive results, without a significance test or confidence
interval. Repeated listeners and related utterances can induce dependence across
pairs. SD measures rating dispersion, not chance-corrected agreement or label
reliability, and depends on the mean and ceiling/floor effects of the bounded
1–5 scale. Greater dispersion alone does not establish why models obtain lower
correlation on accent similarity.

Suggested caption:

> Listener disagreement on the 2,800 training pairs. (a) Empirical cumulative
> distributions of the sample standard deviation of speaker and accent ratings
> across listeners. (b) Paired differences in standard deviation (accent minus
> speaker); positive values indicate greater disagreement about accent. All
> listeners are retained, and every pair receives equal weight.
