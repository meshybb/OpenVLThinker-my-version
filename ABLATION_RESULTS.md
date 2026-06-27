# Ablation Update: GRPO vs. G²RPO vs. FS-G²RPO

## Context

This is not a full reproduction of OpenVLThinkerV2.

Main limitations:
- The final OpenVLThinkerV2 checkpoint is not publicly released.
- The exact filtered OneThinker-600k training subset is not released.
- The original setup requires much larger models, data, and compute than the available university Slurm resources.
Therefore, I implemented a small controlled ablation using the same local setup for all runs.

## Experimental Setup

* Model: `Qwen/Qwen2.5-1.5B-Instruct`
* Dataset: Math12K subset
* Train examples: 256
* Validation examples: 100
* Training steps: 50
* Max response length: 256
* Rollout batch size: 2
* Rollout samples per prompt: 2

## Method Difference

The only changed parameter was the advantage estimator:

* GRPO: `algorithm.adv_estimator=grpo`
* G²RPO: `algorithm.adv_estimator=gs_grpo`
* FS-G²RPO: `algorithm.adv_estimator=fs_gs_grpo`

G²RPO changes how rewards are converted into advantages using a rank-based Gaussian normalization.

After the first comparison, I saw that G²RPO mainly improved answer format and structure, while mathematical accuracy improved only slightly. Based on this, I added a small follow-up variant called FS-G²RPO.

FS-G²RPO applies G²RPO to an adjusted reward:

```text
adjusted_reward = total_reward
                  + 0.5 * format_reward
                  + 0.5 * structure_reward
```

The goal was to test whether emphasizing the reward components that improved most in G²RPO would further strengthen structured answer generation.

## Final Results

| Algorithm | Steps | Train | Val | Validation Reward | Math Accuracy | Format | Structure |
| --------- | ----: | ----: | --: | ----------------: | ------------: | -----: | --------: |
| GRPO      |    50 |   256 | 100 |             0.063 |          0.07 |   0.00 |     0.000 |
| G²RPO     |    50 |   256 | 100 |             0.678 |          0.12 |   0.95 |     0.475 |
| FS-G²RPO  |    50 |   256 | 100 |             0.246 |          0.10 |   0.26 |     0.130 |

## Result Meaning

G²RPO gave the strongest result. Its main improvement was not mathematical accuracy, but much better format and structure reliability.

FS-G²RPO improved format and structure compared with GRPO, but did not outperform the original G²RPO run. This suggests that explicitly adding extra format and structure weight was not better than the original G²RPO normalization in this small setup.

## Qualitative Examples

### Example 1: `ab = 1200`

**Question:** Given `ab = 1200`, where `a` is an integer and `b` is odd, find the largest possible value of `b`.

**Correct answer:** `75`

| Method   | Output                                        | Score |
| -------- | --------------------------------------------- | ----: |
| GRPO     | Long reasoning, but no final `<answer>`       |   0.0 |
| G²RPO    | `<answer>75</answer>`                         |   1.5 |
| FS-G²RPO | Long reasoning, but no final formatted answer |   0.0 |

**Meaning:** G²RPO produced both the correct answer and the required format. FS-G²RPO did not preserve this improvement in this example.

---

### Example 2: Base-2 Geometric Series

**Question:** Find the sum of `0.1₂ - 0.01₂ + 0.001₂ - 0.0001₂ + ...`

**Correct answer:** `1/3`

| Method   | Output                                                     | Score |
| -------- | ---------------------------------------------------------- | ----: |
| GRPO     | Started solving but did not reach a final formatted answer |   0.0 |
| G²RPO    | `<answer>$$\frac{1}{3}$$</answer>`                         |   1.5 |
| FS-G²RPO | Started solving but did not reach a final formatted answer |   0.0 |

**Meaning:** G²RPO gave a short valid answer in the required format, while FS-G²RPO behaved more similarly to GRPO.

---

### Example 3: Rectangle in a Unit Circle

**Question:** A rectangle is inscribed in a unit circle. Find the largest possible area.

**Correct answer:** `2`

| Method   | Output                                         | Score |
| -------- | ---------------------------------------------- | ----: |
| GRPO     | Long incomplete reasoning, no final `<answer>` |   0.0 |
| G²RPO    | `<answer>1</answer>`                           |   0.6 |
| FS-G²RPO | `<answer>$$\frac{1}{2}$$</answer>`             |   0.6 |

**Meaning:** Both G²RPO and FS-G²RPO followed the required answer structure, but both were mathematically wrong. This supports the conclusion that the main improvement is structural, not purely mathematical.

---

## Connection to the G²RPO Formula

The qualitative examples suggest that G²RPO gives a stronger training signal for partial rewards such as **format** and **structure**.

In this experiment, G²RPO did not dramatically improve mathematical correctness. Instead, it made the model much more consistent in producing the required output format:

```text
<thinking>...</thinking>
<answer>...</answer>
```

This matches the algorithmic difference: G²RPO changes how rewards are converted into advantages. As a result, partial reward components can have a clearer effect during training.
FS-G²RPO sometimes preserved the structural behavior, but not consistently. Overall, it did not match the stronger format and structure reliability of the original G²RPO run.


## Main Conclusion

G²RPO mainly improved output organization and format reliability.
FS-G²RPO was a reasonable formula-level follow-up, but in this small Math12K setup it did not improve over the original G²RPO.
