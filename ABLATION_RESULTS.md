# Ablation Update: GRPO vs. G²RPO vs. FS-G²RPO
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

FS-G²RPO is a small follow-up variant motivated by the previous results. Since G²RPO mainly improved answer format and structure, FS-G²RPO applies the same G²RPO normalization to an adjusted reward:

```text
adjusted_reward = total_reward
                  + 0.5 * format_reward
                  + 0.5 * structure_reward
```

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

| Algorithm | Steps | Train | Val | Validation Reward | Math Accuracy | Format | Structure |
| --------- | ----: | ----: | --: | ----------------: | ------------: | -----: | --------: |
| GRPO      |    50 |   256 | 100 |             0.063 |          0.07 |   0.00 |     0.000 |
| G²RPO     |    50 |   256 | 100 |             0.678 |          0.12 |   0.95 |     0.475 |
| FS-G²RPO  |    50 |   256 | 100 |               TBD |           TBD |    TBD |       TBD |


### Result Meaning

G²RPO mainly improved format and structure reliability, while math accuracy improved only slightly.
This means the main improvement is not a large jump in reasoning ability, but a much more stable ability to produce answers in the required format.

---

## Qualitative Examples
The examples below were selected from the saved per-example validation outputs of the final rerun.
### Example 1: `ab = 1200`

**Question:** Given `ab = 1200`, where `a` is an integer and `b` is odd, find the largest possible value of `b`.
**Correct answer:** `75`

| Method | Output                                  | Score |
| ------ | --------------------------------------- | ----: |
| GRPO   | Long reasoning, but no final `<answer>` |   0.0 |
| G²RPO  | `<answer>75</answer>`                   |   1.5 |

**What happened:** G²RPO produced both the correct answer and the required format. GRPO started reasoning but failed to produce a valid final answer.

---

### Example 2: Base-2 Geometric Series

**Question:** Find the sum of `0.1₂ - 0.01₂ + 0.001₂ - 0.0001₂ + ...`
**Correct answer:** `1/3`

| Method | Output                                                     | Score |
| ------ | ---------------------------------------------------------- | ----: |
| GRPO   | Started solving but did not reach a final formatted answer |   0.0 |
| G²RPO  | `<answer>$$\frac{1}{3}$$</answer>`                         |   1.5 |

**What happened:** G²RPO gave a short and valid final answer. This matches the strong improvement in format and structure scores.

---

### Example 3: Rectangle in a Unit Circle

**Question:** A rectangle is inscribed in a unit circle. Find the largest possible area.
**Correct answer:** `2`

| Method | Output                                         | Score |
| ------ | ---------------------------------------------- | ----: |
| GRPO   | Long incomplete reasoning, no final `<answer>` |   0.0 |
| G²RPO  | `<answer>1</answer>`                           |   0.6 |

**What happened:** G²RPO was mathematically wrong, but still received partial reward because it followed the required answer structure. This shows that the improvement is mostly structural, not purely mathematical.

---

## Connection to the G²RPO Formula

The qualitative examples suggest that G²RPO gives a stronger training signal for partial rewards such as **format** and **structure**.

In this experiment, G²RPO did not dramatically improve mathematical correctness. Instead, it made the model much more consistent in producing the required output format:

```text
<thinking>...</thinking>
<answer>...</answer>
```

This matches the algorithmic difference: G²RPO changes how rewards are converted into advantages. As a result, partial reward components can have a clearer effect during training.

**Main conclusion:** G²RPO mainly improved output organization and format reliability. The improvement in math accuracy was smaller.
