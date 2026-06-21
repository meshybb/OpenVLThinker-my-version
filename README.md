## Current Progress

This project is a small-scale controlled experiment based on OpenVLThinker / OpenVLThinkerV2.
The original paper uses a large vision-language model and a filtered training subset that is not fully available, so the goal here is not a full reproduction of the paper. Instead, the goal is to compare GRPO and G²RPO under the same small-scale experimental setup.

So far, I prepared a small mathematical reasoning dataset based on `hiyouga/math12k`.
Two local JSON files were created:

* `data_small/math12k_train_64.json`
* `data_small/math12k_val_50.json`

The dataset was adapted to the EasyR1 reward pipeline by adding the required fields:

* `problem_reserved_text`
* `ground_truth`

I first tested the vision-language model:

* `Qwen/Qwen2.5-VL-3B-Instruct`

This model was chosen because it is related to the Qwen-VL family used in the paper, but smaller than the original model.
Several Slurm smoke tests were performed on L4 GPUs. During this process, I fixed or worked around multiple practical issues:

* replaced `flash_attention_2` with `sdpa` because `flash_attn` was not installed;
* moved from 1 GPU to 2 GPUs because 1×L4 was not enough for the model, FSDP, and vLLM rollout together;
* limited vLLM memory usage using smaller rollout settings;
* set `rollout_batch_size=2` to match the 2-GPU setup;
* disabled `padding_free` to avoid dependency on `flash_attn`;
* disabled actor optimizer offload to avoid CPU/GPU tensor device mismatch during optimizer updates.

The best run so far is:

* `smoke_g2rpo_l4_fix8_2gpu.sh`

This run successfully reached 2/2 G²RPO training steps and produced final validation metrics. However, the Slurm job ended with an OOM kill near the end of execution, most likely due to high CPU/Ray memory usage. Therefore, this run is useful as evidence that the pipeline can run, but it is not stable enough for the final comparison experiment.

## Current Conclusion

`Qwen/Qwen2.5-VL-3B-Instruct` can run a very small G²RPO smoke test on 2×L4 GPUs, but the setup is too memory-heavy and unstable for repeated controlled experiments on the available hardware.

For the final GRPO vs. G²RPO comparison, the next step is to switch to a smaller Qwen instruction model, such as:

* `Qwen/Qwen2.5-1.5B-Instruct`

Although this model is not vision-language, the current dataset is text-based mathematical reasoning, so it is more suitable for a stable controlled comparison.

## Next Tasks

1. Keep the current Qwen2.5-VL-3B smoke test as a hardware feasibility experiment.
2. Create a new smoke-test script for `Qwen/Qwen2.5-1.5B-Instruct`.
3. Run a small G²RPO smoke test with the smaller model.
4. Run a matching GRPO smoke test with the same model, dataset, batch size, and number of steps.
5. If both smoke tests pass, run a slightly longer comparison, for example 10–20 training steps.
6. Collect and compare the following metrics:

   * validation reward score;
   * math accuracy reward;
   * math format reward;
   * training reward;
   * time per step;
   * GPU memory usage;
   * CPU memory usage.
7. Write the final report as a controlled comparison rather than a full reproduction of the original paper.
