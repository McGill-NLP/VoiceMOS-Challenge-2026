#!/usr/bin/env python3
"""Build train / dev-ID / dev-OOD splits from the Track 3 training data.

Since the official dev set ships without labels, we carve our own evaluation
splits out of the labelled `sets/train.csv`. The splits are designed to measure
generalization to unseen systems and unseen utterances:

  * train.csv    - seen systems x seen utterances (the bulk of the data).
  * dev-ID.csv   - In-Distribution: the SAME systems and utterances as train,
                   but held-out sample pairs. Tests generalization to new
                   samples drawn from systems/utterances the model has seen.
  * dev-OOD.csv  - Out-of-Distribution: every sample whose system OR utterance
                   was held out from training. Tests generalization to unseen
                   systems and/or unseen utterances. An extra `ood_type` column
                   tags each row as unseen_system / unseen_utterance /
                   unseen_both.

Splitting is done at the (system_id, utterance_id) sample level. All listener
rows for a sample are kept together in the same split so no pair leaks across
splits.

The wav paths in the CSVs stay relative to the data root (e.g. `wav/...`), so
the output CSVs are drop-in compatible with finetune.py / inference.py as long
as `--data-root` points at the syn distribution directory.
"""
import argparse
import collections
import csv
import json
import os
import random

BASE_HEADER = [
    "system_id", "utterance_id", "listener_id",
    "wav_a_path", "wav_b_path", "spk_sim", "acc_sim",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/vmc2026_track3_train_phase_distro_v3_syn",
                    help="Path to the syn distribution dir (contains sets/ and wav/).")
    ap.add_argument("--train-csv", default=None,
                    help="Source labelled CSV. Defaults to <data-root>/sets/train.csv.")
    ap.add_argument("--outdir", default="data",
                    help="Where to write train.csv / dev-ID.csv / dev-OOD.csv.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    ap.add_argument("--n-ood-systems", type=int, default=3,
                    help="Number of systems held out entirely for OOD.")
    ap.add_argument("--n-ood-utterances", type=int, default=20,
                    help="Number of utterances held out entirely for OOD.")
    ap.add_argument("--dev-id-frac", type=float, default=0.10,
                    help="Fraction of in-distribution samples reserved for dev-ID.")
    return ap.parse_args()


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    src = args.train_csv or os.path.join(args.data_root, "sets", "train.csv")
    with open(src, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows read from {src}")

    systems = sorted({r["system_id"] for r in rows})
    utterances = sorted({r["utterance_id"] for r in rows})
    if args.n_ood_systems >= len(systems):
        raise SystemExit(f"--n-ood-systems ({args.n_ood_systems}) must be < #systems ({len(systems)})")
    if args.n_ood_utterances >= len(utterances):
        raise SystemExit(f"--n-ood-utterances ({args.n_ood_utterances}) must be < #utterances ({len(utterances)})")

    # Choose held-out systems/utterances (sorted samples for deterministic output).
    ood_systems = set(rng.sample(systems, args.n_ood_systems))
    ood_utterances = set(rng.sample(utterances, args.n_ood_utterances))

    # Group listener rows by sample (unique wav_a == (system, utterance)).
    samples = collections.OrderedDict()
    for r in rows:
        samples.setdefault((r["system_id"], r["utterance_id"]), []).append(r)

    train_rows, devid_rows, ood_rows = [], [], []
    ood_counts = collections.Counter()
    id_pool_keys = []

    for key, sample_rows in samples.items():
        sys_id, utt = key
        sys_ood = sys_id in ood_systems
        utt_ood = utt in ood_utterances
        if sys_ood or utt_ood:
            ood_type = ("unseen_both" if sys_ood and utt_ood
                        else "unseen_system" if sys_ood else "unseen_utterance")
            ood_counts[ood_type] += 1
            for r in sample_rows:
                tagged = dict(r)
                tagged["ood_type"] = ood_type
                ood_rows.append(tagged)
        else:
            id_pool_keys.append(key)

    # Split the in-distribution pool into train vs dev-ID at the sample level.
    rng.shuffle(id_pool_keys)
    n_devid = round(len(id_pool_keys) * args.dev_id_frac)
    devid_keys = set(id_pool_keys[:n_devid])
    for key in id_pool_keys:
        (devid_rows if key in devid_keys else train_rows).extend(samples[key])

    # Guarantee dev-ID only contains systems/utterances that remain in train.
    train_systems = {r["system_id"] for r in train_rows}
    train_utts = {r["utterance_id"] for r in train_rows}
    devid_systems = {r["system_id"] for r in devid_rows}
    devid_utts = {r["utterance_id"] for r in devid_rows}
    missing_sys = devid_systems - train_systems
    missing_utt = devid_utts - train_utts
    if missing_sys or missing_utt:
        raise SystemExit(
            "dev-ID contains systems/utterances absent from train "
            f"(systems={sorted(missing_sys)}, utts={sorted(missing_utt)}). "
            "Try a different --seed or a smaller --dev-id-frac."
        )

    os.makedirs(args.outdir, exist_ok=True)
    write_csv(os.path.join(args.outdir, "train.csv"), BASE_HEADER, train_rows)
    write_csv(os.path.join(args.outdir, "dev-ID.csv"), BASE_HEADER, devid_rows)
    write_csv(os.path.join(args.outdir, "dev-OOD.csv"), BASE_HEADER + ["ood_type"], ood_rows)

    manifest = {
        "source_csv": os.path.abspath(src),
        "seed": args.seed,
        "n_ood_systems": args.n_ood_systems,
        "n_ood_utterances": args.n_ood_utterances,
        "dev_id_frac": args.dev_id_frac,
        "ood_systems": sorted(ood_systems),
        "ood_utterances": sorted(ood_utterances),
        "train_systems": sorted(train_systems),
        "counts": {
            "rows": {"train": len(train_rows), "dev_ID": len(devid_rows), "dev_OOD": len(ood_rows)},
            "samples": {
                "train": len(id_pool_keys) - n_devid,
                "dev_ID": n_devid,
                "dev_OOD": sum(ood_counts.values()),
            },
            "dev_OOD_by_type": dict(ood_counts),
        },
    }
    with open(os.path.join(args.outdir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = len(rows)
    print(f"Source: {src}  ({total} listener rows, {len(samples)} samples, "
          f"{len(systems)} systems, {len(utterances)} utterances)")
    print(f"Held-out OOD systems ({len(ood_systems)}):    {', '.join(sorted(ood_systems))}")
    print(f"Held-out OOD utterances ({len(ood_utterances)}): {', '.join(sorted(ood_utterances))}")
    print("-" * 64)
    print(f"{'split':<10}{'samples':>10}{'rows':>10}{'row %':>10}")
    for name, srows, nsamp in [
        ("train", train_rows, len(id_pool_keys) - n_devid),
        ("dev-ID", devid_rows, n_devid),
        ("dev-OOD", ood_rows, sum(ood_counts.values())),
    ]:
        print(f"{name:<10}{nsamp:>10}{len(srows):>10}{100*len(srows)/total:>9.1f}%")
    print("-" * 64)
    print("dev-OOD by type (samples): " +
          ", ".join(f"{k}={v}" for k, v in sorted(ood_counts.items())))
    print(f"Wrote train.csv, dev-ID.csv, dev-OOD.csv, split_manifest.json to {args.outdir}/")


if __name__ == "__main__":
    main()
