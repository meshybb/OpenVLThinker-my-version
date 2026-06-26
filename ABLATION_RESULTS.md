# Ablation Update: GRPO vs. G²RPO

## Context

This is not a full reproduction of OpenVLThinkerV2.

Main limitations:
- The final OpenVLThinkerV2 checkpoint is not publicly released.
- The exact filtered OneThinker-600k training subset is not released.
- The original setup requires much larger models, data, and compute than the available university Slurm resources.

Therefore, I implemented a small controlled ablation that compares GRPO and G²RPO under the same local setup.

## Method Difference

The two runs used the same training pipeline.  
The only changed parameter was the advantage estimator:

- GRPO: `algorithm.adv_estimator=grpo`
- G²RPO: `algorithm.adv_estimator=gs_grpo`

In standard GRPO, rewards are normalized within the sampled responses of the same prompt using the group mean and standard deviation.

In G²RPO, the rewards are first separated by task type and then mapped using a rank-based 1D optimal-transport normalization to Gaussian-like scores.

Since this experiment uses only Math12K text math problems, all validation examples belong to the same task type. Therefore, the practical difference in this ablation is mainly the reward-to-advantage normalization method, not multi-task balancing.


## Experimental Setup

Final setup:

* Model: `Qwen/Qwen2.5-1.5B-Instruct`
* Dataset: Math12K subset
* Train examples: 256
* Validation examples: 100
* Training steps: 50
* Max response length: 256
* Rollout batch size: 2
* Rollout samples per prompt: 2

Before the final run, I also tested a closer Qwen-VL model. It reached a functional G²RPO smoke test, but it was too memory-heavy for stable repeated experiments on the available hardware. Therefore, I used the smaller Qwen instruction model for the final controlled comparison.

## Final Results

Both final runs completed successfully with `ExitCode 0:0`.

| Algorithm | Steps | Train | Val | Validation reward | Math accuracy | Format | Structure |
| --------- | ----: | ----: | --: | ----------------: | ------------: | -----: | --------: |
| GRPO      |    50 |   256 | 100 |             0.135 |          0.15 |   0.00 |     0.000 |
| G²RPO     |    50 |   256 | 100 |             0.162 |          0.16 |   0.03 |     0.015 |


## Result Meaning
G²RPO gave a small positive improvement over GRPO in this controlled run.
The gain is not large enough to claim a strong improvement in mathematical ability. A safer interpretation is that G²RPO slightly improved the training signal and produced a small improvement in answer quality and organization.
Because the experiment used a small model, 50 training steps, 100 validation examples, and one seed, the result should be treated as a limited positive trend rather than strong statistical evidence.
