# VoiceMOS Challenge 2026

Mila-NRC submission for the [VoiceMOS Challenge 2026](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026).

- [track3/](track3/) — speaker & accent similarity prediction.

## Track 3 — the submitted system

**What we submitted is the weak-learner ensemble in [track3/weak/](track3/weak/)**: ridge, SVR and
gradient-boosting regressors fitted on **frozen** embeddings from seven encoders (WavLM Base+/Large,
XLS-R, ECAPA, CommonAccent-ECAPA, ERes2NetV2 x2), pooled as an unweighted mean of the top 16 members
with no fitted parameters. Nothing in it is fine-tuned. For the final run it was refit on train+dev,
with the member list frozen at what the held-out run chose.

The fine-tuned deep models it can be pooled with are in [track3/unified/](track3/unified/).

Utterance-level SRCC:

| system | dev `spk_sim` | dev `acc_sim` | test `spk_sim` | test `acc_sim` |
|---|---|---|---|---|
| deep top-8 (fine-tuned only) | 0.579 | 0.564 | 0.575 | 0.478 |
| **weak top-16 — submitted** | **0.622** | **0.597** | **0.606** | **0.523** |
| deep top-8 + weak top-16 | 0.623 | 0.603 | 0.609 | 0.530 |

dev is held out and the models behind those columns are fitted on train alone; the test columns are
the train+dev refits, scored after the labels were released and used for no selection of any kind.
Packaged predictions are in
[track3/weak/egs/submission_final/](track3/weak/egs/submission_final/), named
`deep<D>-weak<W>_<fitting set>`.

Everything else — encoders, features, the PCA finding, pool-size sweep, member lists, decorrelation
analysis — is in [track3/weak/README.md](track3/weak/README.md) and
[track3/unified/README.md](track3/unified/README.md).
