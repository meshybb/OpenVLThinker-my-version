# Experiment Process Log

This file summarizes the main engineering process, intermediate runs, issues, fixes, and results that led to the final GRPO vs. G²RPO ablation.

## 1. Initial Direction

The project started from OpenVLThinker / OpenVLThinkerV2. The original setup was too large for the available hardware because it uses a large Qwen-VL model, a large filtered dataset, and longer training.

Therefore, I designed a smaller controlled experiment:

* same model
* same data
* same training settings
* same validation setup
* only change: `grpo` vs. `gs_grpo`

## 2. First Model Attempt: Qwen2.5-VL-3B

I first tested:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

This was chosen because it is closer to the Qwen-VL family used in the paper.

The model reached a functional G²RPO smoke test with 2 training steps on 2×L4 GPUs, but it was too memory-heavy for stable repeated experiments. The main issues were:

* 1×L4 was not enough for actor + FSDP + vLLM rollout
* `flash_attn` was unavailable, so I switched to `sdpa`
* `padding_free` required `flash_attn`, so I disabled it
* vLLM memory had to be reduced
* CPU/Ray memory usage was high

Conclusion: the VL model was useful as a feasibility test, but not suitable for the final controlled comparison.

## 3. Final Model Choice

For the final controlled experiment, I switched to:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

This model is smaller and text-only, but the selected dataset is also text-based mathematical reasoning, so it fits the final ablation setup.

## 4. Dataset

I used a subset of:

```text
hiyouga/math12k
```

Initial small subset:

* `math12k_train_64.json`
* `math12k_val_50.json`

Final subset:

* `math12k_train_256.json`
* `math12k_val_100.json`

The dataset was adapted to the EasyR1 reward format with fields such as:

* `problem`
* `answer`
* `ground_truth`
* `problem_reserved_text`
* `task_type=math_reasoning`
* `images=[]`
* `videos=[]`

## 5. Smoke Tests and Intermediate Runs

Both GRPO and G²RPO first passed 2-step smoke tests.

Then I ran 10-step comparisons on the 64/50 subset:

| Algorithm | Steps | Validation reward | Math accuracy | Format |
| --------- | ----: | ----------------: | ------------: | -----: |
| G²RPO     |    10 |             0.696 |          0.20 |   0.86 |
| GRPO      |    10 |             0.756 |          0.28 |   0.84 |

Then I ran clean 20-step comparisons after fixing disk-space issues:

| Algorithm | Steps | Validation reward | Math accuracy | Format | Structure |
| --------- | ----: | ----------------: | ------------: | -----: | --------: |
| GRPO      |    20 |             0.690 |          0.38 |   0.58 |      0.29 |
| G²RPO     |    20 |             0.798 |          0.34 |   0.82 |      0.41 |

## 6. Practical Issues Fixed

Main issues solved during the project:

* Replaced `flash_attention_2` with `sdpa`
* Disabled `padding_free`
* Used 2 GPUs instead of 1 GPU
* Reduced vLLM memory pressure
* Fixed missing dataset fields
* Corrected `task_type` from `math` to `math_reasoning`
* Added `HYDRA_FULL_ERROR=1`
* Added `RAY_DEDUP_LOGS=0`
* Cleaned large checkpoint weights while preserving experiment logs

The disk issue was important: checkpoints filled the home directory, so I removed large model weight folders while keeping:

* `experiment_log.jsonl`
* `generations.log`
* `experiment_config.json`

## 7. Final Ablation

The final ablation used:

* 256 training examples
* 100 validation examples
* 50 training steps
* same model and settings for both methods

Final results:

| Algorithm | Steps | Train | Val | Validation Reward | Math Accuracy | Format | Structure |
| --------- | ----: | ----: | --: | ----------------: | ------------: | -----: | --------: |
| GRPO      |    50 |   256 | 100 |             0.063 |          0.07 |   0.00 |     0.000 |
| G²RPO     |    50 |   256 | 100 |             0.678 |          0.12 |   0.95 |     0.475 |

## 8. Final Takeaway

G²RPO substantially improved format and structure reliability, while mathematical accuracy improved only slightly.

The experiment is not a full reproduction of the original paper. It is a small-scale engineering reproduction and ablation under limited university compute.


## 9. Connection to the G²RPO Formula

The qualitative examples suggest that G²RPO provides a clearer training signal for partial reward components such as **format** and **structure**.

Compared with GRPO, G²RPO changes how rewards are converted into advantages. In this small experiment, that seems to help the model consistently learn the required output pattern:

```text
<thinking>...</thinking>
<answer>...</answer>
```

Therefore, the main conclusion is:

**G²RPO improves output organization and format reliability more clearly than mathematical correctness.**
The math accuracy improves only slightly, but the format and structure rewards improve substantially.

## 10. Additional Variant: Format/Structure-Aware G²RPO

The final GRPO vs. G²RPO ablation showed that the main improvement of G²RPO was in output format and structure, while mathematical accuracy improved only slightly.

Based on this observation, I added a small follow-up variant called **FS-G²RPO**.

The idea is to keep the same G²RPO Gaussian rank-based advantage normalization, but slightly increase the influence of the reward components that were most improved in the previous experiment: format and structure.

Instead of applying G²RPO directly on the total reward, the new variant uses:

```text
adjusted_reward = total_reward
                  + 0.5 * format_reward
                  + 0.5 * structure_reward
```

Then the same G²RPO normalization is applied to this adjusted reward.

This experiment uses the same setup as the final ablation

The goal is not to improve the model significantly, but to test a small formula-level change that is motivated by the previous results.
