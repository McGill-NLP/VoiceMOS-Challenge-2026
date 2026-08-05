#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-accent
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Unified sweep for Track 3 ACCENT similarity (acc_sim).
#
#   sbatch track3/jobs/voicemos-track3-unified-accent.sh
#
# A thin wrapper: it pins the target and hands off to
# voicemos-track3-unified-sweep.sh, which holds the ladder, the environment setup and the
# scoring. Deliberately NOT a copy of that script -- the existing rank-n-contrast pair are
# 256-line duplicates differing in four lines, which means every fix has to be applied
# twice. Everything except METRICS still passes through, so the sweep's own overrides work
# here unchanged:
#
#   sbatch --export=ALL,ENCODER=commonaccent-ecapa \
#       track3/jobs/voicemos-track3-unified-accent.sh
#   sbatch --export=ALL,CONFIGS="base corn coral" \
#       track3/jobs/voicemos-track3-unified-accent.sh
#
# Runs one target x 6 default arms = 6 experiments. Measured on an L40S with ECAPA:
# base / moe / corn / stack 29 min each at 8,000 steps, freeze 25, stack-rnc 28 at 4,000
# steps (epoch-matched, see the sweep) -> about 2.8 h with ECAPA or CommonAccent, which is
# why this asks for 6 h rather than the combined job's 12 h.
# ERes2NetV2 is roughly 3x slower per optimizer step and needs ~8 h: pass
# --time=12:00:00.
#
# Best acc_sim results so far, for picking --encoder (see ../../BRANCHES.md):
#   commonaccent-ecapa ft-lr1e-4   UTT-SRCC 0.448  SYS-SRCC 0.914
#   eres2netv2         ft-lr1e-4   UTT-SRCC 0.456  SYS-SRCC 0.908   <- best utterance-level
#   ecapa-voxceleb     lr1e-3      UTT-SRCC 0.407  SYS-SRCC 0.928   <- best system-level
#
# Worth knowing before choosing: a VoxCeleb speaker-ID encoder is trained to be INVARIANT
# to accent, which is why the zero-shot baseline emits identical predictions for both
# targets. commonaccent-ecapa is discriminative in exactly this dimension, and dev.yj-v2
# reached its best accent number (UTT-SRCC 0.502) by switching to it for acc_sim only.

export METRICS=acc_sim

# Resolved from REPO, not from $BASH_SOURCE: sbatch stages the submitted script to a spool
# directory on the node, so the script's own path is not the repository path at run time.
REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
SWEEP="$REPO/track3/jobs/voicemos-track3-unified-sweep.sh"
[ -f "$SWEEP" ] || { echo "ERROR: cannot find $SWEEP (set REPO=... if the checkout moved)"; exit 1; }
exec bash "$SWEEP"
