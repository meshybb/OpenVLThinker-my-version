#!/bin/bash
#SBATCH --job-name=smoke_g2rpo_f8
#SBATCH --output=smoke_g2rpo_f8_%j.out
#SBATCH --error=smoke_g2rpo_f8_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12

set -x

cd /home/dsi/baruchm9/Seminary/new/OpenVLThinker/EasyR1

eval "$(conda shell.bash hook)"
conda activate easyr1_clean

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$PWD:$PYTHONPATH

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

python3 -m verl.trainer.main \
  config=examples/config_g2rpo.yaml \
  data.train_files=data_small/math12k_train_64.json \
  data.test_files="['data_small/math12k_val_50.json']" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  data.video_key=videos \
  data.image_dir=null \
  data.val_image_dir=null \
  data.rollout_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.min_pixels=3136 \
  data.max_pixels=262144 \
  data.filter_overlong_prompts=false \
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
  worker.rollout.max_num_batched_tokens=1024 \
  algorithm.adv_estimator=gs_grpo \
  algorithm.online_filtering=false \
  trainer.logger="['console','file']" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.max_steps=2 \
  trainer.val_freq=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.experiment_name=smoke_g2rpo_fix8_2gpu_qwen25vl3b_math12k
