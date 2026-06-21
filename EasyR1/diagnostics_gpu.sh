#!/bin/bash
set -x

echo "===== GPU NODE BASIC ====="
hostname
date
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "===== GPU INFO ====="
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader || true

echo "===== CPU/RAM ====="
lscpu | head -40 || true
free -h || true
df -h .
df -h $HOME

echo "===== PYTORCH CUDA TEST ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, round(p.total_memory / 1024**3, 2), "GB")
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("matmul ok:", y.shape, y.dtype, y.device)
PY

echo "===== VLLM / TRANSFORMERS IMPORT TEST ====="
python - <<'PY'
mods = ["transformers", "datasets", "accelerate", "vllm", "ray", "flash_attn", "qwen_vl_utils"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", "no_version"))
    except Exception as e:
        print(m, "FAIL", repr(e))
PY
