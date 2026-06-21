#!/bin/bash
set -x

echo "===== BASIC ====="
hostname
whoami
date
pwd

echo "===== DISK / QUOTA ====="
df -h .
df -h $HOME
quota -s || true
du -sh ~/.cache/huggingface 2>/dev/null || true
du -sh . 2>/dev/null || true

echo "===== SLURM ACCOUNTS / PARTITIONS ====="
sacctmgr list assoc where user=$USER format=User,Account -nP 2>/dev/null || true
sinfo -o "%P %a %D %l %G %m %N" || true
squeue -u $USER || true

echo "===== MODULES / CONDA / PYTHON ====="
module list 2>&1 || true
module avail cuda 2>&1 | head -80 || true
which python || true
python --version || true
which python3 || true
python3 --version || true
which conda || true
conda info --envs 2>/dev/null || true

echo "===== CUDA / NVIDIA COMMANDS ====="
which nvidia-smi || true
nvidia-smi || true
which nvcc || true
nvcc --version || true

echo "===== PROJECT STRUCTURE ====="
ls -lah
find . -maxdepth 3 -type f | sort | head -200

echo "===== IMPORTANT FILES ====="
ls -lah examples || true
ls -lah local_scripts || true
sed -n '1,220p' examples/config_g2rpo.yaml 2>/dev/null || true
sed -n '1,220p' local_scripts/run_g2rpo_rl_slurm.sh 2>/dev/null || true

echo "===== CHECK GRPO / G2RPO CODE ====="
grep -R "GS_GRPO\|gs_grpo\|compute_pertask_gaussian\|AdvantageEstimator.GS" -n verl examples local_scripts 2>/dev/null || true
grep -R "adv_estimator" -n examples local_scripts verl/trainer 2>/dev/null || true
grep -R "reward_function" -n examples local_scripts verl 2>/dev/null | head -80 || true

echo "===== PYTHON PACKAGES ====="
python - <<'PY'
import sys, importlib.util
print("python:", sys.version)
mods = ["torch", "transformers", "datasets", "accelerate", "vllm", "ray", "flash_attn", "qwen_vl_utils"]
for m in mods:
    spec = importlib.util.find_spec(m)
    print(f"{m}: {'FOUND' if spec else 'MISSING'}")
PY
