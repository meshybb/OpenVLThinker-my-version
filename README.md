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







## Small-Scale GRPO vs. G²RPO Experiment

After the initial Qwen2.5-VL-3B feasibility test, I switched to a smaller model for the controlled comparison experiment:

* `Qwen/Qwen2.5-1.5B-Instruct`

This model was selected because the original Qwen3-VL-Instruct-8B model and the exact training data from the paper are too large or unavailable for the current university server setup. Although this model is not vision-language, the selected Math12K subset is text-based, so it is suitable for a controlled GRPO vs. G²RPO comparison.

I created and used the following Slurm script:

* `EasyR1/run_math12k_rl_small.sh`

The script supports both GRPO and G²RPO using the same model, dataset, batch settings, response length, and validation setup. The only main difference between the two runs is the advantage estimator:

* GRPO: `algorithm.adv_estimator=grpo`
* G²RPO: `algorithm.adv_estimator=gs_grpo`

The experiment used:

* model: `Qwen/Qwen2.5-1.5B-Instruct`
* train data: `data_small/math12k_train_64.json`
* validation data: `data_small/math12k_val_50.json`
* GPUs: 2×L4
* max response length: 256
* rollout batch size: 2
* rollout samples per prompt: 2

Both algorithms first passed a 2-step smoke test. Then, I ran a 10-step controlled comparison.

### 10-Step Results

| Model                 | Algorithm | Steps | Validation reward | Math accuracy reward | Math format reward | Time per step | CPU memory | GPU memory |
| --------------------- | --------: | ----: | ----------------: | -------------------: | -----------------: | ------------: | ---------: | ---------: |
| Qwen2.5-1.5B-Instruct |     G²RPO |    10 |             0.696 |                 0.20 |               0.86 |      128.954s |   56.121GB |    4.976GB |
| Qwen2.5-1.5B-Instruct |      GRPO |    10 |             0.756 |                 0.28 |               0.84 |      116.381s |    56.23GB |    4.976GB |

In this small-scale setup, GRPO achieved a higher validation reward and higher math accuracy reward, while G²RPO achieved a slightly higher format reward. Since this experiment uses a very small dataset subset and only 10 training steps, the result should not be interpreted as a general conclusion that GRPO is better than G²RPO. It only describes the behavior observed under this specific constrained setup.

## Updated Next Tasks

1. Commit the current experiment script and README update.
2. Optionally run a longer controlled comparison with 20 training steps for both G²RPO and GRPO.
3. Keep the setup identical between the two methods.
4. Compare validation reward, math accuracy reward, format reward, runtime, and memory usage.
5. In the final seminar report, present this as a small-scale controlled comparison, not as a full reproduction of the original OpenVLThinkerV2 paper.


G2RPO 20-step run on A100 reached 20/20 training steps, saved a global_step_20 checkpoint, and logged validation metrics at step 20. The Slurm job was marked as FAILED with ExitCode 1 after completion, so this run should be reported cautiously, but the step-20 validation metrics are available in experiment_log.jsonl.
math12k_gs_grpo_qwen25_15b_steps20_resp256_16879570





```markdown
## Small-Scale GRPO vs. G²RPO Experiment

After the initial Qwen2.5-VL-3B feasibility test, I switched to a smaller model for the controlled comparison experiment:

* `Qwen/Qwen2.5-1.5B-Instruct`

This model was selected because the original Qwen3-VL-Instruct-8B model and the exact training data from the paper are too large or unavailable for the current university server setup. Although this model is not vision-language, the selected Math12K subset is text-based, so it is suitable for a controlled GRPO vs. G²RPO comparison.

The dataset used for the controlled experiment is a small subset of `hiyouga/math12k`:

* `data_small/math12k_train_64.json` — 64 training examples
* `data_small/math12k_val_50.json` — 50 validation examples

The same model, dataset, batch settings, response length, and validation setup were used for both algorithms. The only main difference between the two runs was the advantage estimator:

* GRPO: `algorithm.adv_estimator=grpo`
* G²RPO: `algorithm.adv_estimator=gs_grpo`

The experiment was run using:

* model: `Qwen/Qwen2.5-1.5B-Instruct`
* GPUs: 2 GPUs
* max response length: 256
* rollout batch size: 2
* rollout samples per prompt: 2
* validation frequency: every 5 steps

Both algorithms first passed 2-step smoke tests. Then, I ran a 10-step controlled comparison. After fixing a disk-space issue, I reran a cleaner 20-step comparison.

### 10-Step Preliminary Results

| Model | Algorithm | Steps | Validation reward | Math accuracy reward | Math format reward | Time per step | CPU memory | GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | G²RPO | 10 | 0.696 | 0.20 | 0.86 | 128.954s | 56.121GB | 4.976GB |
| Qwen2.5-1.5B-Instruct | GRPO | 10 | 0.756 | 0.28 | 0.84 | 116.381s | 56.230GB | 4.976GB |

These 10-step runs were useful as preliminary clean runs showing that both algorithms could run successfully in the EasyR1/OpenVLThinker pipeline.

### Disk-Space Issue and Fix

During the first 20-step attempts, the runs reached step 20 and logged validation metrics, but some Slurm jobs were marked as failed at the end. The main issue was that the home directory was full:

* `/home` reached 100% usage
* checkpoint files occupied about 75GB
* the system reported `No space left on device`

To fix this, I deleted only the large model checkpoint `.pt` files under `checkpoints/*/global_step_*/actor/`, while keeping the important experiment logs:

* `experiment_log.jsonl`
* `generations.log`
* `experiment_config.json`

After this cleanup, available disk space increased to about 56GB and the 20-step runs completed successfully.

I also added the following debugging environment variables to the Slurm script:

* `HYDRA_FULL_ERROR=1`
* `RAY_DEDUP_LOGS=0`

These make future errors easier to debug.

### Clean 20-Step Results

After fixing the disk-space issue, I reran both GRPO and G²RPO for 20 steps using the same setup. Both runs completed successfully with `ExitCode 0:0`.

| Model | Algorithm | Steps | Validation reward | Math accuracy reward | Math format reward | Math structure reward | Time per step | CPU memory | GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | GRPO | 20 | 0.690 | 0.38 | 0.58 | 0.29 | 174.491s | 55.118GB | 4.976GB |
| Qwen2.5-1.5B-Instruct | G²RPO | 20 | 0.798 | 0.34 | 0.82 | 0.41 | 155.526s | 56.933GB | 4.976GB |

In the clean 20-step comparison, G²RPO achieved a higher overall validation reward, higher format reward, and higher structure reward. GRPO achieved a slightly higher math accuracy reward. This suggests that, in this small-scale setup, G²RPO improved the combined reward by producing outputs that better matched the expected format and structure, while GRPO had slightly better mathematical accuracy.

This result should still be interpreted carefully because the experiment uses a small dataset subset and a small number of training steps. It is not a full reproduction of OpenVLThinkerV2 and does not prove that one method is generally better. It only reports the behavior observed under this controlled, resource-constrained setup.

## Current Conclusion

The initial Qwen2.5-VL-3B smoke test showed that the original vision-language direction was possible but too memory-heavy for stable repeated experiments on the available hardware.

The controlled Qwen2.5-1.5B experiment is stable and suitable for the seminar ablation. The clean 20-step comparison shows that both GRPO and G²RPO can be run successfully under the same EasyR1/OpenVLThinker training pipeline.

The most important practical issue found during the experiments was disk space. Large checkpoint files filled the home directory and caused some runs to fail at the end despite reaching the final training step. After deleting unnecessary `.pt` checkpoint files and keeping only logs/metrics, the runs completed successfully.

## Next Tasks

1. Create a larger Math12K subset for the final ablation, for example:
   * 256 training examples
   * 100 validation examples
2. Run the final controlled comparison with:
   * GRPO, 50 steps
   * G²RPO, 50 steps
3. Keep the setup identical between both methods:
   * same model
   * same dataset
   * same rollout settings
   * same max response length
   * same validation frequency
4. Compare:
   * validation reward score
   * math accuracy reward
   * math format reward
   * math structure reward
   * runtime
   * CPU and GPU memory usage
5. Present the final report as a small-scale controlled ablation, not as a full reproduction of the original OpenVLThinkerV2 paper.
```
