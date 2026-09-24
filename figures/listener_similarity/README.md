# Speaker and accent ratings by listener

From the repository root:

```bash
python3 -m pip install -r figures/requirements.txt
python3 figures/listener_similarity/plot_listener_similarity.py
```

The script reads the official training CSV by default. Its paths are relative to
the script, so it can run from any working directory. Override the input or output
location with `--csv PATH` or `--output-dir PATH`.

Outputs:

- `listener_similarity.pdf`: vector figure for the paper.
- `listener_similarity.png`: 300 dpi preview.
- `listener_similarity.svg`: editable vector figure.
- `listener_similarity.csv`: exact per-listener statistics.

The plot shows Spearman correlation between each listener's paired speaker and
accent ratings, ordered by decreasing correlation and labeled by listener ID.
Rating counts and identical-score percentages remain in the CSV statistics,
but are not plotted. Correlation is marked undefined when
either dimension is constant. No listeners are removed, and ratings are not
averaged across listeners. These are descriptive statistics, without confidence
intervals.

High correlation means that a listener ranks the two dimensions similarly; it
does not necessarily mean they assign identical scores. Correlation alone
establishes that a listener confuses speaker and accent, or explains why accent
similarity is harder for models to predict. Listener groups can also differ in
which samples they rated.

The figure uses a compact height for one-column placement: 4.3 by 4.0 inches
in the script and 6.5 by 6.0 inches in the LaTeX notebook. Widths are unchanged;
smaller listener labels and markers keep the 25 rows readable.

Suggested caption:

> Association between speaker and accent similarity ratings for each of the 25
> training listeners, measured by within-listener Spearman correlation.
> Listeners are ordered by correlation and labeled by ID.
