#!/bin/bash
#SBATCH --job-name=math12k_rl
#SBATCH --output=math12k_rl_%j.out
#SBATCH --error=math12k_rl_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=12

set -euo pipefail
set -x

cd /home/dsi/baruchm9/Seminary/new/OpenVLThinker/EasyR1

eval "$(conda shell.bash hook)"
conda activate easyr1_clean

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gs_grpo}
STEPS=${STEPS:-2}
MAX_RESP=${MAX_RESP:-256}
VAL_FREQ=${VAL_FREQ:-1}

EXP_NAME="math12k_${ADV_ESTIMATOR}_qwen25_15b_steps${STEPS}_resp${MAX_RESP}_${SLURM_JOB_ID}"

python3 -m verl.trainer.main \
  config=examples/config_g2rpo.yaml \
  data.train_files=data_small/math12k_train_256.json \
  data.test_files="['data_small/math12k_val_100.json']" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  data.video_key=videos \
  data.image_dir=null \
  data.val_image_dir=null \
  data.rollout_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=${MAX_RESP} \
  data.min_pixels=3136 \
  data.max_pixels=262144 \
  data.filter_overlong_prompts=false \
  data.seed=1 \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.actor.model.trust_remote_code=true \
  worker.actor.global_batch_size=1 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.padding_free=false \
  worker.actor.optim.lr=2e-6 \
  worker.actor.optim.strategy=adamw_bf16 \
  worker.actor.fsdp.enable_cpu_offload=true \
  worker.actor.offload.offload_params=true \
  worker.actor.offload.offload_optimizer=false \
  worker.ref.fsdp.enable_cpu_offload=true \
  worker.ref.offload.offload_params=true \
  worker.ref.padding_free=false \
  worker.rollout.n=2 \
  worker.rollout.tensor_parallel_size=2 \
  worker.rollout.max_model_len=1024 \
  worker.rollout.gpu_memory_utilization=0.45 \
  worker.rollout.enforce_eager=true \
  worker.rollout.temperature=1.0 \
  worker.rollout.top_p=1.0 \
  worker.rollout.seed=1 \
  worker.rollout.max_num_batched_tokens=2048 \
  worker.rollout.val_override_config.temperature=0.01 \
  worker.rollout.val_override_config.top_p=0.001 \
  worker.rollout.val_override_config.top_k=1 \
  worker.rollout.val_override_config.n=1 \
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  algorithm.online_filtering=false \
  trainer.logger="['console','file']" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.max_steps=${STEPS} \
  trainer.val_freq=${VAL_FREQ} \
  trainer.val_before_train=false \
  trainer.save_freq=${STEPS} \
  trainer.save_limit=1 \
  trainer.val_generations_to_log=100 \
  trainer.find_last_checkpoint=false \
  trainer.save_checkpoint_path=checkpoints/${EXP_NAME} \
  trainer.experiment_name=${EXP_NAME}


  # ---------- Save compact final results ----------
RESULT_TAG=${RESULT_TAG:-$ADV_ESTIMATOR}
FINAL_DIR="final_results/${RESULT_TAG}_${SLURM_JOB_ID}"

mkdir -p "$FINAL_DIR"

echo "Saving compact results to: $FINAL_DIR"

# Copy validation samples and important logs if they exist
find . -path "*${EXP_NAME}*" -type f \( \
  -name "validation_samples_step_${STEPS}.jsonl" -o \
  -name "experiment_log.jsonl" -o \
  -name "generations.log" -o \
  -name "experiment_config.json" \
\) -exec cp -v {} "$FINAL_DIR"/ \; || true

# Extra fallback: search by job id, in case the trainer saved under a slightly different path
find . -path "*${SLURM_JOB_ID}*" -type f -name "validation_samples_step_${STEPS}.jsonl" \
  -exec cp -v {} "$FINAL_DIR"/ \; || true

echo "Final result files:"
ls -lh "$FINAL_DIR" || true