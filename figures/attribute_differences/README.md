# Does the model recover differences between speaker and accent similarity?

From the repository root, after installing `figures/requirements.txt`:

```bash
python3 figures/attribute_differences/plot_attribute_differences.py
```

For the same 600 official test pairs, plot the human difference
`accent MOS - speaker MOS` on the x-axis against the model difference
`predicted accent - predicted speaker` on the y-axis. The default model is the
submitted weak-16 ensemble, fitted on train+dev. Test labels are used only for
this descriptive analysis of the fixed predictions.

The axes have equal scales and symmetric limits. The diagonal represents exact
recovery of both direction and magnitude. Points near the horizontal zero line
indicate small predicted differences, even when human differences are large.
Same-sign quadrants indicate recovery of the direction of a nonzero human gap.
Hexagons count pairs on a logarithmic color scale; there is no jitter, smoothing,
or fitted trend line. All 600 pairs are included, including 140 human ties.

The figure annotations report:

- **Gap SRCC:** Spearman correlation between observed and predicted signed
  differences (0.378). This differs from correlation between the two targets
  within human ratings or within model predictions.
- **Gap MAE:** mean absolute error of the predicted difference (0.347 MOS points).
  For context, always predicting a zero difference gives MAE 0.364.
- **Direction agreement:** percentage of the 460 pairs with nonzero human gaps
  for which the predicted gap has the same sign (67.4%, or 310/460). A zero
  predicted gap counts as incorrect on these pairs. Human ties are excluded
  from this statistic only, and no extra minimum-gap threshold is imposed.

Differences are rounded to 12 decimal places before computing statistics so
floating-point subtraction does not split mathematical ties in the human MOS.
Mean absolute gaps are 0.364 for human ratings and 0.136 for model predictions.
The model partly recovers the ordering and direction of the differences but
produces a narrower range of gaps. This is descriptive evidence, not a
significance test or proof of a causal explanation. Human MOS differences are
also noisy estimates; small differences should not be treated as certain
perceptual distinctions.

Outputs are `attribute_differences.pdf`, `.png` (300 dpi), and `.svg`, plus
`attribute_differences.csv` (aligned scores, differences, and errors for each
pair) and `summary.json` (exact statistics and input paths).

Input validation and alignment are reused from
`../human_vs_model/plot_human_vs_model.py`; no generated output from that script
is required. Pairs are joined on both waveform paths, with duplicates, missing
or extra pairs, and invalid scores rejected. Options are `--labels`,
`--speaker-predictions`, `--accent-predictions`, `--model-label`, and
`--output-dir`. Default paths are relative to the script.

Suggested caption:

> Recovery of speaker–accent differences on 600 test pairs. The horizontal axis
> shows human accent MOS minus speaker MOS, and the vertical axis shows the
> corresponding difference between submitted weak-16 predictions. The dashed
> diagonal indicates exact recovery; hexagons show pair counts on a logarithmic
> scale. Direction agreement is computed on the 460 pairs with nonzero human
> differences, while SRCC and MAE use all pairs.
