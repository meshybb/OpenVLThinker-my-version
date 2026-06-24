# OpenVLThinker Small-Scale Ablation

## Original Project

This work is based on the original OpenVLThinker repository:

https://github.com/uclanlp/OpenVLThinker

My repository contains a small-scale controlled ablation built on top of the original project, adapted to the available university Slurm resources.

## Goal

The goal was to compare two reinforcement learning advantage estimators under the same setup:

* GRPO
* G²RPO, implemented in the codebase as `gs_grpo`

The only intended difference between the two final runs was:

```bash
algorithm.adv_estimator=grpo
algorithm.adv_estimator=gs_grpo
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

| Algorithm | Steps | Train | Val | Validation reward | Math accuracy | Format | Structure |
| --------- | ----: | ----: | --: | ----------------: | ------------: | -----: | --------: |
| GRPO      |    50 |   256 | 100 |             0.135 |          0.15 |   0.00 |     0.000 |
| G²RPO     |    50 |   256 | 100 |             0.162 |          0.16 |   0.03 |     0.015 |

## Conclusion

In the final controlled ablation, G²RPO achieved a small but consistent improvement over GRPO across the main validation metrics.

This experiment should not be interpreted as a full reproduction of OpenVLThinkerV2. It is a small-scale controlled ablation showing that the training pipeline works locally and that G²RPO slightly outperformed GRPO under the same constrained setup.

For the full experiment history, failed attempts, fixes, and intermediate runs, see `EXPERIMENT_PROCESS.md`.

For a short summary intended for the seminar update, see `ABLATION_RESULTS.md`.
