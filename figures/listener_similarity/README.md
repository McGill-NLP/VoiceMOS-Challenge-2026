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

Panel (a) shows Spearman correlation between each listener's paired speaker and
accent ratings. Panel (b) shows the percentage of pairs assigned exactly the same
score on both dimensions. Both panels order listeners by decreasing correlation;
labels include the number of rated pairs. Correlation is marked undefined when
either dimension is constant. No listeners are removed, and ratings are not
averaged across listeners. These are descriptive statistics, without confidence
intervals.

High correlation means that a listener ranks the two dimensions similarly; it
does not necessarily mean they assign identical scores. Neither measure alone
establishes that a listener confuses speaker and accent, or explains why accent
similarity is harder for models to predict. Listener groups can also differ in
which samples they rated.

Suggested caption:

> Association between speaker and accent similarity ratings for each of the 25
> training listeners: (a) within-listener Spearman correlation and (b) percentage
> of identical scores. Listeners are ordered by correlation; numbers in
> parentheses indicate the number of rated pairs.
