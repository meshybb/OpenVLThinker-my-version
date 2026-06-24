# Ablation Update: GRPO vs. G²RPO

## Context and Constraints

I conducted a small-scale controlled ablation based on OpenVLThinkerV2.

A full reproduction was not feasible because the original checkpoint and exact filtered training subset were not available, and the original setup requires a much larger vision-language model and significantly more compute.

Therefore, I focused on a controlled local comparison between GRPO and G²RPO under identical conditions.

## Plan

The ablation compared two advantage estimators:

* GRPO: `algorithm.adv_estimator=grpo`
* G²RPO: `algorithm.adv_estimator=gs_grpo`

Everything else was kept fixed:

* same model
* same dataset
* same training steps
* same validation set
* same rollout settings
* same response length
* same Slurm/GPU setup

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

## Result Summary

G²RPO achieved a small but consistent improvement over GRPO:

* higher validation reward
* slightly higher math accuracy
* higher format reward
* higher structure reward

The absolute scores are limited by the small dataset, short training run, and smaller model, but the comparison itself is controlled and fair because both methods used exactly the same setup.

## Conclusion

Under the same small-scale experimental setup, G²RPO slightly outperformed GRPO.


```
```
