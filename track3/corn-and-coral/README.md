# VoiceMOS Challenge 2026 - Track 3

Our experiments for Track 3: predicting speaker and accent similarity between a synthetic sample and a reference. For the official baseline system, see [BASELINES.md](BASELINES.md).

## Data splits

The official dev set ships without labels (they will be released on July 31), so we create our own evaluation splits from the labelled `sets/train.csv` to measure generalization to **unseen systems and listeners**. [build_splits.py](build_splits.py) writes three files to `data/`:

- `train.csv` — seen systems/listeners (~75%)
- `dev-ID.csv` — In-Distribution: same systems/listeners, held-out cells (~8%)
- `dev-OOD.csv` — Out-of-Distribution: held-out systems and/or listeners (~17%)

Splitting is at the `(system_id, listener_id)` level, so no cell straddles splits.

```bash
python build_splits.py
```

## Models
