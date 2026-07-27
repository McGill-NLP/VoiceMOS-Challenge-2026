#!/usr/bin/env python3
"""Build train / dev-ID / dev-OOD splits from the Track 3 training data.

Since the official dev set ships without labels, we carve our own evaluation
splits out of the labelled `sets/train.csv`. The splits are designed to measure
generalization to unseen systems and unseen listeners:

  * train.csv    - seen systems x seen listeners (the bulk of the data).
  * dev-ID.csv   - In-Distribution: the SAME systems and listeners as train,
                   but held-out (system, listener) cells. Tests generalization
                   to new ratings from systems/listeners the model has seen.
  * dev-OOD.csv  - Out-of-Distribution: every rating whose system OR listener
                   was held out from training. Tests generalization to unseen
                   systems and/or unseen listeners. An extra `ood_type` column
                   tags each row as unseen_system / unseen_listener /
                   unseen_both.

Splitting is done at the (system_id, listener_id) cell level. All rows of a cell
are kept together in the same split so a system/listener combination never
straddles splits.

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
    ap.add_argument("--n-ood-systems", type=int, default=2,
                    help="Number of systems held out entirely for OOD.")
    ap.add_argument("--n-ood-listeners", type=int, default=2,
                    help="Number of listeners held out entirely for OOD.")
    ap.add_argument("--dev-id-frac", type=float, default=0.10,
                    help="Fraction of in-distribution cells reserved for dev-ID.")
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
    listeners = sorted({r["listener_id"] for r in rows})
    if args.n_ood_systems >= len(systems):
        raise SystemExit(f"--n-ood-systems ({args.n_ood_systems}) must be < #systems ({len(systems)})")
    if args.n_ood_listeners >= len(listeners):
        raise SystemExit(f"--n-ood-listeners ({args.n_ood_listeners}) must be < #listeners ({len(listeners)})")

    # Choose held-out systems/listeners (from sorted lists for deterministic output).
    ood_systems = set(rng.sample(systems, args.n_ood_systems))
    ood_listeners = set(rng.sample(listeners, args.n_ood_listeners))

    # Group rows by (system, listener) cell.
    cells = collections.OrderedDict()
    for r in rows:
        cells.setdefault((r["system_id"], r["listener_id"]), []).append(r)

    train_rows, devid_rows, ood_rows = [], [], []
    ood_counts = collections.Counter()
    id_pool_keys = []

    for key, cell_rows in cells.items():
        sys_id, lis_id = key
        sys_ood = sys_id in ood_systems
        lis_ood = lis_id in ood_listeners
        if sys_ood or lis_ood:
            ood_type = ("unseen_both" if sys_ood and lis_ood
                        else "unseen_system" if sys_ood else "unseen_listener")
            ood_counts[ood_type] += 1
            for r in cell_rows:
                tagged = dict(r)
                tagged["ood_type"] = ood_type
                ood_rows.append(tagged)
        else:
            id_pool_keys.append(key)

    # Split the in-distribution pool into train vs dev-ID at the cell level.
    rng.shuffle(id_pool_keys)
    n_devid = round(len(id_pool_keys) * args.dev_id_frac)
    devid_keys = set(id_pool_keys[:n_devid])
    for key in id_pool_keys:
        (devid_rows if key in devid_keys else train_rows).extend(cells[key])

    # Guarantee dev-ID only contains systems/listeners that remain in train.
    train_systems = {r["system_id"] for r in train_rows}
    train_listeners = {r["listener_id"] for r in train_rows}
    devid_systems = {r["system_id"] for r in devid_rows}
    devid_listeners = {r["listener_id"] for r in devid_rows}
    missing_sys = devid_systems - train_systems
    missing_lis = devid_listeners - train_listeners
    if missing_sys or missing_lis:
        raise SystemExit(
            "dev-ID contains systems/listeners absent from train "
            f"(systems={sorted(missing_sys)}, listeners={sorted(missing_lis)}). "
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
        "n_ood_listeners": args.n_ood_listeners,
        "dev_id_frac": args.dev_id_frac,
        "ood_systems": sorted(ood_systems),
        "ood_listeners": sorted(ood_listeners),
        "train_systems": sorted(train_systems),
        "counts": {
            "rows": {"train": len(train_rows), "dev_ID": len(devid_rows), "dev_OOD": len(ood_rows)},
            "cells": {
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
    print(f"Source: {src}  ({total} listener rows, {len(cells)} cells, "
          f"{len(systems)} systems, {len(listeners)} listeners)")
    print(f"Held-out OOD systems ({len(ood_systems)}):   {', '.join(sorted(ood_systems))}")
    print(f"Held-out OOD listeners ({len(ood_listeners)}): {', '.join(sorted(ood_listeners))}")
    print("-" * 64)
    print(f"{'split':<10}{'cells':>10}{'rows':>10}{'row %':>10}")
    for name, srows, ncell in [
        ("train", train_rows, len(id_pool_keys) - n_devid),
        ("dev-ID", devid_rows, n_devid),
        ("dev-OOD", ood_rows, sum(ood_counts.values())),
    ]:
        print(f"{name:<10}{ncell:>10}{len(srows):>10}{100*len(srows)/total:>9.1f}%")
    print("-" * 64)
    print("dev-OOD by type (cells): " +
          ", ".join(f"{k}={v}" for k, v in sorted(ood_counts.items())))
    print(f"Wrote train.csv, dev-ID.csv, dev-OOD.csv, split_manifest.json to {args.outdir}/")


if __name__ == "__main__":
    main()
