#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-ssl-sizing
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=david.guzman@mila.quebec

# Step time and peak GPU memory for the three SSL encoders, so the grid jobs can be sized
# from measurements instead of guesses. Runs 12 real training steps per configuration on the
# actual data, unfrozen (the expensive phase), and prints a table.
#
#   sbatch track3/jobs/ssl/voicemos-track3-ssl-sizing.sh

module load miniconda/3; module load gcc/9.3.0; module load cuda/12.3.2
export HF_HOME=$SCRATCH/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
conda activate VoiceMOS
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH
cd "${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}/track3/unified" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv

python - <<'PY'
import time, torch, logging, csv, os, random
import torchaudio
logging.basicConfig(level=logging.WARNING)
from model import UnifiedModel

DR = "../baseline/data/vmc2026_track3_train_phase_distro_v3_syn"
rows = list(csv.DictReader(open(f"{DR}/sets/train_plus_dev.csv")))
random.seed(0); random.shuffle(rows)

def load(p):
    w, sr = torchaudio.load(os.path.join(DR, p))
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    return w[0]

pool = [(load(r["wav_a_path"]), load(r["wav_b_path"])) for r in rows[:64]]

def batch(n, i):
    a = [pool[(i + k) % len(pool)][0] for k in range(n)]
    b = [pool[(i + k) % len(pool)][1] for k in range(n)]
    la = torch.tensor([x.shape[0] for x in a]); lb = torch.tensor([x.shape[0] for x in b])
    pa = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
    pb = torch.nn.utils.rnn.pad_sequence(b, batch_first=True)
    return pa.cuda(), pb.cuda(), la.cuda(), lb.cuda()

print(f"\n{'encoder':<22}{'batch':>6}{'s/step':>9}{'peak GiB':>10}{'20k steps':>12}")
print("-" * 59)
for enc in ("wavlm-base-plus-l4", "wavlm-large-l4", "xlsr-300m-l4"):
    for bs in (4, 8, 16):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            m = UnifiedModel(encoder_name=enc, target_metric="spk_sim", objective="coral",
                             head_type="moe", interaction="bilinear").cuda().train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            for i in range(12):
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                wa, wb, la, lb = batch(bs, i * bs)
                out = m(wa, wb, la, lb)
                loss = sum(v.float().mean() for v in out.values()
                           if torch.is_tensor(v) and v.dtype.is_floating_point)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 10
            gib = torch.cuda.max_memory_allocated() / 2**30
            # 20,000 optimizer steps at effective batch 16
            hrs = dt * 20000 * (16 / bs) / 3600
            print(f"{enc:<22}{bs:>6}{dt:>9.3f}{gib:>10.1f}{hrs:>10.1f} h")
        except torch.cuda.OutOfMemoryError:
            print(f"{enc:<22}{bs:>6}{'OOM':>9}")
        finally:
            del m, opt; torch.cuda.empty_cache()
PY
