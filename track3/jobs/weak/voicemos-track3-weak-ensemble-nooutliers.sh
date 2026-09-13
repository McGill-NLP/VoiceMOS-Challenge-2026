#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-weak-ensemble-nooutliers
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# The submitted weak-learner ensemble, end to end, fitted on the OUTLIER-REMOVED training set
# and scored on held-out dev AND test.
#
#   sbatch track3/jobs/weak/voicemos-track3-weak-ensemble-nooutliers.sh
#
# Companion to voicemos-track3-weak-ensemble-train.sh: same encoders, same features, same
# learners, same FROZEN top-16 member list. The only change is the fitting set,
# sets/train-without-outliers.csv, which removes two whole LISTENERS (25 -> 23, 1,097 of 13,687
# rating rows) and keeps all 2,800 pairs, so only the per-pair mean targets move.
#
# WHAT THIS ADDS OVER voicemos-track3-weak-outliers.sh. That job fitted the same 84 models into
# egs/weak_outliers/ but only compared members on dev. This one packages the weak top-16 as a
# submission and scores it on test as well, next to the system we actually submitted.
#
# BOTH EVALUATION SETS ARE HELD OUT. dev is not in the fitting set, so its numbers compare
# directly with weak-16 [train] (0.622 / 0.597 uSRCC). Test labels were released after the
# challenge closed: they are REPORT-ONLY here. Members are frozen from the original held-out
# run and nothing below selects, weights or tunes anything on test.
#
# COMPARISONS PRINTED AT THE END
#   dev   no-outliers vs weak-16 [train]                  (train+dev is in-sample on dev, omitted)
#   test  no-outliers vs weak-16 [train] and [train+dev]  (the latter is the submitted system)
# each with all six metrics and a paired bootstrap CI on the uSRCC difference.
#
# NOTHING EXISTING IS OVERWRITTEN. Models go to a new tree and the submission to a new folder;
# the job refuses to write into a submission folder that already holds predictions unless
# OVERWRITE=1. Pass OUTDIR=egs/weak_outliers to reuse the earlier fits instead of refitting --
# train_weak.py is resumable and skips completed runs, so that finishes in seconds.
#
# Deliberately NOT using `set -e`: one failing learner family must not kill the sweep.

START_TIME=$SECONDS
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST   cpus: ${SLURM_CPUS_PER_TASK:-?}"

##################################################################
# Environment
##################################################################
module load miniconda/3
module load gcc/9.3.0
module load cuda/12.3.2

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

conda activate VoiceMOS
if [ "$CONDA_DEFAULT_ENV" != "VoiceMOS" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected VoiceMOS"; exit 1
fi
python -c "import torch, torchaudio, sklearn, scipy, speechbrain" \
    || { echo "ERROR: torch/torchaudio/sklearn/scipy/speechbrain not importable"; exit 1; }

# torchaudio >= 2.9 dispatches load() to torchcodec, which dies with
# "libnppicc.so.12: cannot open shared object file" unless the NVIDIA libs that ship with
# torch are on the loader path. Only stage 1 loads audio, and only if the cache is incomplete.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
cd "$REPO/track3/weak" || exit 1

##################################################################
# Configuration
##################################################################
TRAIN_CSV=${TRAIN_CSV:-../baseline/data/vmc2026_track3_train_phase_distro_v3_syn/sets/train-without-outliers.csv}
FEATDIR=${FEATDIR:-egs/features}
OUTDIR=${OUTDIR:-egs/weak_ens_nooutliers}
SUBDIR=${SUBDIR:-egs/submission_final}
TAG=${TAG:-train_no_outliers}
SUB="$SUBDIR/deep0-weak16_$TAG"

# The submitted system, for the side-by-side. Fitted on train (dev held out) and train+dev.
REF_TRAIN=${REF_TRAIN:-egs/submission_final/deep0-weak16_train_only}
REF_TRAINDEV=${REF_TRAINDEV:-egs/submission_final/deep0-weak16_train_plus_dev}

SV_ENCODERS=${SV_ENCODERS:-"sv:ecapa-voxceleb sv:commonaccent-ecapa sv:eres2netv2 sv:eres2netv2-w24s4ep4"}
SSL_ENCODERS=${SSL_ENCODERS:-"ssl:WAVLM_BASE_PLUS ssl:WAVLM_LARGE ssl:WAV2VEC2_XLSR_300M"}
SSL_LAYERS=${SSL_LAYERS:-"4 8 -1"}

[ -f "$TRAIN_CSV" ] || { echo "ERROR: no training csv at $TRAIN_CSV"; exit 1; }
for d in "$REF_TRAIN" "$REF_TRAINDEV"; do
    [ -d "$d" ] || { echo "ERROR: reference submission $d missing"; exit 1; }
done
if ls "$SUB"/test_*.csv >/dev/null 2>&1 && [ "${OVERWRITE:-0}" != "1" ]; then
    echo "ERROR: $SUB already holds predictions. Refusing to overwrite; set OVERWRITE=1 or TAG=..."
    exit 1
fi

echo "=================================================================="
echo "weak-learner ensemble, fitted on the OUTLIER-REMOVED training set"
echo "  train  : $TRAIN_CSV"
echo "  feats  : $FEATDIR   (7 encoders, SSL layers $SSL_LAYERS, no PCA)"
echo "  models : $OUTDIR"
echo "  sub    : $SUB"
echo "  compare: $REF_TRAIN  |  $REF_TRAINDEV"
echo "=================================================================="

##################################################################
# Stage 1 -- cache the frozen encoder features (GPU, ~10 min, skipped if present)
#
# No new audio is involved: every wav the outlier-removed set references is already cached.
##################################################################
echo ""; echo "### stage 1: feature cache ($(date))"
python extract_features.py \
    --encoders $SV_ENCODERS $SSL_ENCODERS \
    --ssl-layers $SSL_LAYERS \
    --outdir "$FEATDIR"
if [ $? -ne 0 ]; then echo "FEATURE EXTRACTION FAILED"; exit 1; fi
E=$((SECONDS - START_TIME)); echo "[features ready, $((E/3600))h $((E%3600/60))m in]"

##################################################################
# Stage 2 -- fit the weak learners (CPU, ~22 min)
#
# Exactly what the 16 frozen members need: ridge on all 13 feature files, and ksvr + hgb on the
# four speaker/accent-ID encoders. Grouped CV on system_id selects hyperparameters as before.
##################################################################
echo ""; echo "### stage 2a: ridge, all feature files ($(date))"
python train_weak.py --train-csv "$TRAIN_CSV" --pca-threshold 99999 \
    --features "$FEATDIR/*.npz" --feature-sets full compact \
    --learners ridge --outdir "$OUTDIR"
[ $? -ne 0 ] && echo "RIDGE SWEEP FAILED"

echo ""; echo "### stage 2b: ksvr + hgb, speaker/accent-ID encoders ($(date))"
python train_weak.py --train-csv "$TRAIN_CSV" --pca-threshold 99999 \
    --features "$FEATDIR/ecapa-voxceleb.npz" "$FEATDIR/commonaccent-ecapa.npz" \
               "$FEATDIR/eres2netv2.npz" "$FEATDIR/eres2netv2-w24s4ep4.npz" \
    --feature-sets full compact --learners ksvr hgb --outdir "$OUTDIR"
[ $? -ne 0 ] && echo "SV SWEEP FAILED"

E=$((SECONDS - START_TIME)); echo "[fitting done, $((E/3600))h $((E%3600/60))m in]"

if grep -q '"dev_in_train": true' "$OUTDIR/stage1_manifest.jsonl" 2>/dev/null; then
    echo "ERROR: $OUTDIR has records flagged dev_in_train -- dev would not be held out. Stopping."
    exit 1
fi

##################################################################
# Stage 3 -- package the weak top-16 from the FROZEN member list
##################################################################
echo ""; echo "### stage 3: submission ($(date))"
for T in spk_sim acc_sim; do
    python make_submission.py --target "$T" --deep-k 0 --weak-k 16 \
        --weak-dir "$OUTDIR/preds" --out "$SUB" \
        --note "weak top-16 only, fitted on $TRAIN_CSV (2 listeners removed), no PCA; members frozen from the held-out train run"
    [ $? -ne 0 ] && { echo "SUBMISSION FAILED for $T"; exit 1; }
done

##################################################################
# Stage 4 -- dev and test scores, next to the submitted system
#
# Report-only on test: every composition below was fixed before this job ran.
##################################################################
echo ""
echo "=================================================================="
echo "no-outliers weak-16 vs the submitted weak-16, held-out dev and test"
echo "=================================================================="
python - "$SUB" "$REF_TRAIN" "$REF_TRAINDEV" "$OUTDIR" <<'PY'
import csv, json, os, sys
import numpy as np, scipy.stats

sub, ref_train, ref_traindev, outdir = sys.argv[1:5]
EV = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets"
LABELS = {"dev": f"{EV}/dev_with_labels.csv", "test": f"{EV}/vmc2026_track3_test_with_labels.csv"}
sys.path.insert(0, ".")
from make_submission import WEAK_MEMBERS


def labels(split, metric):
    rows = list(csv.DictReader(open(LABELS[split])))
    keys = [(r["wav_a_path"], r["wav_b_path"]) for r in rows]
    return ({k: i for i, k in enumerate(keys)}, np.array([float(r[metric]) for r in rows]),
            np.array([r["system_id"] for r in rows]))


def load(path, metric, kidx):
    v = np.full(len(kidx), np.nan)
    for r in csv.DictReader(open(path)):
        i = kidx.get((r["wav_a_path"], r["wav_b_path"]))
        if i is not None:
            v[i] = float(r[f"pred_{metric}"])
    if np.isnan(v).any():
        raise SystemExit(f"{path}: {int(np.isnan(v).sum())} pairs without a prediction")
    return v


def six(t, p, sysid):
    st, sp = {}, {}
    for s, a, b in zip(sysid, t, p):
        st.setdefault(s, []).append(a); sp.setdefault(s, []).append(b)
    x = np.array([np.mean(st[k]) for k in st]); y = np.array([np.mean(sp[k]) for k in st])
    return [float(q) for q in (np.mean((t - p) ** 2), scipy.stats.pearsonr(t, p).statistic,
            scipy.stats.spearmanr(t, p).statistic, np.mean((x - y) ** 2),
            scipy.stats.pearsonr(x, y).statistic, scipy.stats.spearmanr(x, y).statistic)]


def boot_diff(t, a, b, seed=0, n_boot=2000):
    """Paired bootstrap over utterances: uSRCC(b) - uSRCC(a)."""
    rng, n, d = np.random.default_rng(seed), len(t), []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        d.append(scipy.stats.spearmanr(t[s], b[s]).statistic
                 - scipy.stats.spearmanr(t[s], a[s]).statistic)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


NAMES = ["uMSE", "uLCC", "uSRCC", "sMSE", "sLCC", "sSRCC"]
summary = {"fitting_set": "train-without-outliers.csv", "members": "frozen weak top-16",
           "test_is_report_only": True, "scores": {}, "uSRCC_delta_vs": {}, "determinism_check": {}}

for metric in ("spk_sim", "acc_sim"):
    summary["scores"][metric], summary["uSRCC_delta_vs"][metric] = {}, {}
    for split in ("dev", "test"):
        kidx, t, sysid = labels(split, metric)
        systems = {"weak-16 [train]": load(f"{ref_train}/{split}_{metric}.csv", metric, kidx)}
        if split == "test":   # on dev the train+dev refit is in-sample, so it is not comparable
            systems["weak-16 [train+dev] (submitted)"] = load(f"{ref_traindev}/test_{metric}.csv", metric, kidx)
        systems["weak-16 [no-outliers]"] = load(f"{sub}/{split}_{metric}.csv", metric, kidx)

        print(f"\n### {metric}, {split} ({len(t)} pairs, {len(set(sysid))} systems)")
        print(f"  {'system':<34}" + "".join(f"{c:>8}" for c in NAMES))
        print("  " + "-" * 82)
        summary["scores"][metric][split] = {}
        for name, p in systems.items():
            s = six(t, p, sysid)
            summary["scores"][metric][split][name] = dict(zip(NAMES, [round(q, 4) for q in s]))
            print(f"  {name:<34}" + "".join(f"{q:>8.3f}" for q in s))

        new = systems["weak-16 [no-outliers]"]
        summary["uSRCC_delta_vs"][metric][split] = {}
        for name, p in systems.items():
            if name == "weak-16 [no-outliers]":
                continue
            m, lo, hi = boot_diff(t, p, new)
            summary["uSRCC_delta_vs"][metric][split][name] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            print(f"  no-outliers minus {name:<32} uSRCC {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # The earlier voicemos-track3-weak-outliers.sh run fitted the same configurations. The
    # learners are deterministic (fixed random_state), so the members should match exactly.
    old = "egs/weak_outliers/preds"
    if os.path.abspath(outdir) != os.path.abspath("egs/weak_outliers") and os.path.isdir(old):
        kidx, _, _ = labels("test", metric)
        diffs = [np.abs(load(f"{outdir}/preds/{r}__{metric}__test.csv", metric, kidx)
                        - load(f"{old}/{r}__{metric}__test.csv", metric, kidx)).max()
                 for r in WEAK_MEMBERS[metric]]
        summary["determinism_check"][metric] = float(max(diffs))
        print(f"  determinism: max |pred - egs/weak_outliers pred| over the 16 members = {max(diffs):.2e}")

with open(f"{sub}/scores.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nscores -> {sub}/scores.json")
PY
[ $? -ne 0 ] && echo "SCORING FAILED"

echo ""
echo "Reference: on held-out dev the earlier outliers run found uSRCC -0.002 (spk_sim) and"
echo "-0.003 (acc_sim) for weak-16, with system-level MSE improving (acc_sim 0.040 -> 0.032)."
echo "Judge a test difference by its CI, not by the point estimate: n=600."

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
