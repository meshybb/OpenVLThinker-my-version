#!/bin/bash
set -x

echo "===== HF ENV ====="
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

echo "===== HF CACHE SIZE ====="
du -sh ~/.cache/huggingface 2>/dev/null || true
find ~/.cache/huggingface -maxdepth 3 -type d 2>/dev/null | head -80 || true

echo "===== INTERNET / HF TEST ====="
python - <<'PY'
try:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info("Qwen/Qwen2.5-VL-3B-Instruct")
    print("HF model access OK:", info.modelId)
except Exception as e:
    print("HF model access FAIL:", repr(e))

try:
    from datasets import load_dataset
    ds = load_dataset("hiyouga/math12k", split="train[:3]")
    print("dataset access OK:", ds)
    print(ds[0])
except Exception as e:
    print("dataset access FAIL:", repr(e))
PY
