# Final Checkpoint Weights

The final actor checkpoints were saved on the BIU Slurm server but were not committed to Git because each actor checkpoint is approximately 21GB.

## GRPO final checkpoint

Path:

`/home/dsi/baruchm9/Seminary/new/OpenVLThinker/EasyR1/checkpoints/math12k_grpo_qwen25_15b_steps50_resp256_16883195/global_step_50/actor`

## G²RPO final checkpoint

Path:

`/home/dsi/baruchm9/Seminary/new/OpenVLThinker/EasyR1/checkpoints/math12k_gs_grpo_qwen25_15b_steps50_resp256_16883201/global_step_50/actor`

## Saved outputs

The per-example validation outputs are committed under:

- `EasyR1/final_results/grpo_16883195/validation_samples_step_50.jsonl`
- `EasyR1/final_results/g2rpo_16883201/validation_samples_step_50.jsonl`

These files include prompt, model output, ground truth, total score, correctness score, format score, structure score, and length score for each validation example.