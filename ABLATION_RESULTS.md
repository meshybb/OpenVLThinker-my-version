# Ablation Results: GRPO vs. G²RPO

## Project Goal

This project is a small-scale controlled experiment based on OpenVLThinker / OpenVLThinkerV2.

The original paper uses a large vision-language model, a large filtered training dataset, and substantially more compute. Since the released resources and the available university hardware do not allow a full reproduction of the original setup, I focused on a controlled ablation experiment.

The goal of the ablation was to compare two advantage estimators under the same local experimental conditions:

* GRPO
* G²RPO, implemented in the codebase as `gs_grpo`

## What I Changed

The main ablation variable was the advantage estimator:

* GRPO run: `algorithm.adv_estimator=grpo`
* G²RPO run: `algorithm.adv_estimator=gs_grpo`

Everything else was kept the same between the two final runs:

* same model
* same dataset
* same number of training steps
* same validation set
* same rollout settings
* same maximum response length
* same Slurm/GPU setup

## Experimental Setup

Because the full OpenVLThinkerV2 setup was too large for the available hardware, I used a smaller text-based setup:

* Model: `Qwen/Qwen2.5-1.5B-Instruct`
* Dataset: subset of `hiyouga/math12k`
* Training examples: 256
* Validation examples: 100
* Training steps: 50
* Validation frequency: every 10 steps
* Max response length: 256
* Rollout batch size: 2
* Rollout samples per prompt: 2
* GPU setup: 2 GPUs on the university Slurm cluster

The dataset was adapted to the EasyR1 reward pipeline by adding the required fields, including:

* `problem_reserved_text`
* `ground_truth`
* `task_type = math_reasoning`

## Final Results

Both final runs completed successfully with `ExitCode 0:0`.

| Model                 | Algorithm | Steps | Train | Validation | Validation reward | Math accuracy reward | Math format reward | Math structure reward | Time per step | CPU memory | GPU memory |
| --------------------- | --------: | ----: | ----: | ---------: | ----------------: | -------------------: | -----------------: | --------------------: | ------------: | ---------: | ---------: |
| Qwen2.5-1.5B-Instruct |      GRPO |    50 |   256 |        100 |             0.135 |                 0.15 |               0.00 |                 0.000 |      315.801s |   55.551GB |    4.976GB |
| Qwen2.5-1.5B-Instruct |     G²RPO |    50 |   256 |        100 |             0.162 |                 0.16 |               0.03 |                 0.015 |      312.260s |   57.119GB |    4.976GB |

## Main Finding

In the final controlled ablation, G²RPO achieved a small but consistent improvement over GRPO across the main validation metrics.

Compared to GRPO, G²RPO achieved:

* higher validation reward: 0.162 vs. 0.135
* slightly higher math accuracy reward: 0.16 vs. 0.15
* higher format reward: 0.03 vs. 0.00
* higher structure reward: 0.015 vs. 0.000

However, the absolute scores were still low for both methods. This means that the experiment does not show strong overall performance, but it does show that under this small-scale controlled setup, G²RPO performed slightly better than GRPO.

## Interpretation

The result suggests that G²RPO may help stabilize the reward signal compared to standard GRPO in this setup, especially with respect to the combined validation reward and output-format-related metrics.

At the same time, both methods struggled with the task. The low format and structure rewards indicate that the model often failed to produce answers in the expected structure required by the reward function.

Therefore, the main conclusion is:

> Under the same small-scale experimental setup, G²RPO achieved a small but consistent improvement over GRPO, but both methods remained limited by the small dataset, the smaller text-only model, and the short training run.

## Limitations

This experiment should not be interpreted as a full reproduction of OpenVLThinkerV2.

The main limitations are:

1. Dataset limitation
   The experiment used only 256 training examples and 100 validation examples from Math12K, rather than the large filtered dataset used in the original paper.

2. Model limitation
   The experiment used `Qwen/Qwen2.5-1.5B-Instruct`, which is much smaller than the original model and is not a vision-language model.

3. Compute limitation
   The experiment used only 50 training steps with a small rollout batch size on limited university GPU resources.

4. Task limitation
   The final setup used text-based mathematical reasoning, not the full multimodal setting of the original OpenVLThinkerV2 paper.

## Conclusion

This ablation provides a controlled local comparison between GRPO and G²RPO. The final 50-step experiment showed that G²RPO slightly outperformed GRPO under identical conditions, but the overall performance remained low due to the constrained experimental setup.

The experiment is therefore best presented as a small-scale ablation and engineering reproduction of the training pipeline, rather than as a full reproduction of the original OpenVLThinkerV2 results.
