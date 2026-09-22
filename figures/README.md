# Paper figures

Each analysis has its own folder containing its script, documentation, data
summaries, and generated PDF/PNG/SVG files. Shared dependencies are in
`requirements.txt`.

From the repository root:

```bash
python3 -m pip install -r figures/requirements.txt
python3 figures/listener_similarity/plot_listener_similarity.py
python3 figures/listener_disagreement/plot_listener_disagreement.py
python3 figures/human_vs_model/plot_human_vs_model.py
python3 figures/attribute_differences/plot_attribute_differences.py
```

- [Listener similarity](listener_similarity/README.md): association between each
  listener's speaker and accent ratings, and percentage of identical scores.
- [Listener disagreement](listener_disagreement/README.md): rating dispersion
  across listeners for each waveform pair, comparing speaker and accent.
- [Human versus model](human_vs_model/README.md): speaker–accent association in
  human MOS and submitted weak-16 predictions for the same test pairs.
- [Attribute differences](attribute_differences/README.md): recovery of the
  direction and magnitude of human accent-minus-speaker MOS differences.

The first two scripts use individual training ratings, retain all listeners,
and accept `--csv PATH`. The last two scripts use test MOS and fixed
test predictions, with separate input flags documented in their folders. All
scripts accept `--output-dir PATH`. Default paths are relative to the scripts,
so they also work from another working directory.
