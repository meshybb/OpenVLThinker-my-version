#!/bin/bash

project_name='openvlthinkerv2-rl'

MODEL_PATH="Qwen3/Qwen3-VL-8B-Instruct"
TRAIN_FILE="example_data/sample_train_data.json" 

# Training data image dir: 
IMAGE_DIR="todo"  #please update here

TEST_FILE_MATH="example_data/sample_math_val_data.json"
TEST_FILE_GENERAL_VQA="example_data/sample_generalVQA_val_data.json"

VAL_IMAGE_DIR="todo" # your validation image dir 
TEST_FILES="['${TEST_FILE_MATH}', '${TEST_FILE_GENERAL_VQA}']"

CHEKCPOINT_DIR="./checkpoints" # todo, update to save to your local dir 

ROLLOUT_BS=512 
GLOBAL_BS=128  
MB_PER_UPDATE=1 
MB_PER_EXP=2  
TP_SIZE=1 
N_GPUS_PER_NODE=8
NNODES=1 

exp_name="qwen3vl8b_gaussian_grpo_${ROLLOUT_BS}_gb${GLOBAL_BS}_${SLURM_JOB_ID}_${MB_PER_UPDATE}_${MB_PER_EXP}"
CHEKCPOINT_PATH="${CHEKCPOINT_DIR}/${exp_name}"

export WANDB_API_KEY='your api key'
export WANDB_KEY='your api key'
export wandb_name=$exp_name

python3 -m verl.trainer.main \
    config=examples/config_g2rpo.yaml \
    data.train_files=${TRAIN_FILE} \
    data.test_files=\"${TEST_FILES}\" \
    data.image_dir=${IMAGE_DIR} \
    data.val_image_dir=${VAL_IMAGE_DIR} \
    data.rollout_batch_size=${ROLLOUT_BS} \
    worker.actor.global_batch_size=${GLOBAL_BS} \
    worker.actor.micro_batch_size_per_device_for_update=${MB_PER_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MB_PER_EXP} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=2e-6 \
    worker.rollout.tensor_parallel_size=${TP_SIZE} \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.filter_key=accuracy \
    algorithm.adv_estimator=gs_grpo \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=100 \
    trainer.save_checkpoint_path=${CHEKCPOINT_PATH}


