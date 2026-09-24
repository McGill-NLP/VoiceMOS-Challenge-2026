# Paper figures

Each analysis has its own folder containing its script, documentation, data
summaries, and generated PDF/PNG/SVG files. Shared dependencies are in
`requirements.txt`.

## LaTeX figures in Google Colab

[Figures.ipynb](Figures.ipynb) contains a section for each of the four figures,
using LaTeX-rendered serif text and bold labels styled after
`Results_Latex_seeingculture.ipynb`. Open it in Colab and run setup to clone
`https://github.com/McGill-NLP/VoiceMOS-Challenge-2026.git` on `dev.dg/figures`.
The plotting CSVs are loaded directly from that branch. Each section exports
PDF, PNG, SVG, and summary statistics; the final
cell downloads the results together as a ZIP. The human-versus-model figure
keeps its statistics in notebook output rather than on the plot.

The notebook installs LaTeX only in Colab. A local preview is available by
setting `USE_TEX = False`; the current local checkout is used and must be on
`dev.dg/figures`.

## Local scripts

From the repository root:

```bash
python3 -m pip install -r figures/requirements.txt
python3 figures/listener_similarity/plot_listener_similarity.py
python3 figures/listener_disagreement/plot_listener_disagreement.py
python3 figures/human_vs_model/plot_human_vs_model.py
python3 figures/attribute_differences/plot_attribute_differences.py
```

- [Listener similarity](listener_similarity/README.md): association between each
  listener's speaker and accent ratings, shown as Spearman correlation.
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
