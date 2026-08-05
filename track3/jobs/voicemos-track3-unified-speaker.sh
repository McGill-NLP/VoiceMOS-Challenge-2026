#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-speaker
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Unified sweep for Track 3 SPEAKER similarity (spk_sim).
#
#   sbatch track3/jobs/voicemos-track3-unified-speaker.sh
#
# A thin wrapper: it pins the target and hands off to
# voicemos-track3-unified-sweep.sh, which holds the ladder, the environment setup and the
# scoring. Deliberately NOT a copy of that script -- the existing rank-n-contrast pair are
# 256-line duplicates differing in four lines, which means every fix has to be applied
# twice. Everything except METRICS still passes through, so the sweep's own overrides work
# here unchanged:
#
#   sbatch --export=ALL,ENCODER=eres2netv2 --time=12:00:00 \
#       track3/jobs/voicemos-track3-unified-speaker.sh
#   sbatch --export=ALL,CONFIGS="base corn coral" \
#       track3/jobs/voicemos-track3-unified-speaker.sh
#
# Runs one target x 6 default arms = 6 experiments. Measured on an L40S with ECAPA:
# base / moe / corn / stack 29 min each at 8,000 steps, freeze 25, stack-rnc 28 at 4,000
# steps (epoch-matched, see the sweep) -> about 2.8 h with ECAPA or CommonAccent, which is
# why this asks for 6 h rather than the combined job's 12 h.
# ERes2NetV2 is roughly 3x slower per optimizer step and needs ~8 h: pass
# --time=12:00:00 as above.
#
# Best spk_sim results so far, for picking --encoder (see ../../BRANCHES.md):
#   eres2netv2         ft-lr1e-4   UTT-SRCC 0.521  SYS-SRCC 0.883   <- best utterance-level
#   ecapa-voxceleb     lr1e-3      UTT-SRCC 0.458  SYS-SRCC 0.945   <- best system-level
#   commonaccent-ecapa ft-lr1e-4   UTT-SRCC 0.503  SYS-SRCC 0.900

export METRICS=spk_sim

# Resolved from REPO, not from $BASH_SOURCE: sbatch stages the submitted script to a spool
# directory on the node, so the script's own path is not the repository path at run time.
REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
SWEEP="$REPO/track3/jobs/voicemos-track3-unified-sweep.sh"
[ -f "$SWEEP" ] || { echo "ERROR: cannot find $SWEEP (set REPO=... if the checkout moved)"; exit 1; }
exec bash "$SWEEP"
