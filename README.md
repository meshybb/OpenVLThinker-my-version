# OpenVLThinker Small-Scale Ablation

## Original Project

This work is based on the original OpenVLThinker repository:

https://github.com/uclanlp/OpenVLThinker

My repository contains a small-scale controlled ablation built on top of the original project, adapted to the available university Slurm resources.

## Goal

The goal was to compare two reinforcement learning advantage estimators under the same setup:

* GRPO
* G²RPO, implemented in the codebase as `gs_grpo`
* FS-G²RPO, a small formula-level variant implemented as `fs_gs_grpo`

The only intended difference between the two final runs was:

```bash
algorithm.adv_estimator=grpo
algorithm.adv_estimator=gs_grpo
algorithm.adv_estimator=fs_gs_grpo
```

## Setup

The final experiment used:

* Model: `Qwen/Qwen2.5-1.5B-Instruct`
* Dataset: subset of `hiyouga/math12k`
* Train set: 256 examples
* Validation set: 100 examples
* Training steps: 50
* Max response length: 256
* Rollout batch size: 2
* Rollout samples per prompt: 2
* GPU setup: 2 GPUs on the university Slurm cluster

The dataset was adapted to the EasyR1 reward pipeline by adding the required fields, including `ground_truth`, `problem_reserved_text`, and `task_type=math_reasoning`.

## Final Results

Both final runs completed successfully with `ExitCode 0:0`.

| Algorithm | Steps | Train | Val | Validation Reward | Math Accuracy | Format | Structure |
| --------- | ----: | ----: | --: | ----------------: | ------------: | -----: | --------: |
| GRPO      |    50 |   256 | 100 |             0.063 |          0.07 |   0.00 |     0.000 |
| G²RPO     |    50 |   256 | 100 |             0.678 |          0.12 |   0.95 |     0.475 |
| FS-G²RPO  |    50 |   256 | 100 |             0.246 |          0.10 |   0.26 |     0.130 |

## FS-G²RPO Variant

The first GRPO vs. G²RPO comparison showed that G²RPO mainly improved answer **format** and **structure**, while mathematical accuracy improved only slightly.

Based on this observation, I added a small follow-up variant called **FS-G²RPO**.

FS-G²RPO keeps the same G²RPO Gaussian rank-based normalization, but applies it to an adjusted reward:

```text
adjusted_reward = total_reward
                  + 0.5 * format_reward
                  + 0.5 * structure_reward
```

The goal is to test whether emphasizing the reward components that G²RPO already improved can further strengthen structured answer generation.


## Conclusion

In the final controlled ablation, G²RPO achieved a small but consistent improvement over GRPO across the main validation metrics.

This experiment should not be interpreted as a full reproduction of OpenVLThinkerV2. 
It is a small-scale controlled ablation showing that the training pipeline works locally and that G²RPO substantially improved format and structure reliability, while mathematical accuracy improved only slightly.

For the full experiment history, failed attempts, fixes, and intermediate runs, see [EXPERIMENT_PROCESS.md](EXPERIMENT_PROCESS.md).

For a short summary intended for the seminar update, see [ABLATION_RESULTS.md](ABLATION_RESULTS.md).
The final saved outputs are available under:

* [GRPO final outputs](EasyR1/final_results/grpo_16883195/)
* [G²RPO final outputs](EasyR1/final_results/g2rpo_16883201/)

The final actor checkpoint locations are documented in:

* [Final checkpoint weights location](EasyR1/final_results/WEIGHTS_LOCATION.md)


## Connection to the G²RPO Formula

The qualitative examples suggest that G²RPO provides a clearer training signal for partial reward components such as **format** and **structure**.

Compared with GRPO, G²RPO changes how rewards are converted into advantages. In this small experiment, that seems to help the model consistently learn the required output pattern:

```text
<thinking>...</thinking>
<answer>...</answer>
```

Therefore, the main conclusion is:

**G²RPO improves output organization and format reliability more clearly than mathematical correctness.**
The math accuracy improves only slightly, but the format and structure rewards improve substantially.
