# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF

import math
import torch
from collections import defaultdict

if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GDPO = "gdpo"
    GS_GDPO = "gs_gdpo"
    GS_GRPO = "gs_grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """Decorator to register a advantage estimator function with a given name."""

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage and return for a given advantage estimator."""
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GS_GRPO) 
def compute_pertask_gaussian_outcome_advantage_grpo(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    index: list | torch.Tensor, 
    problem_type: list | torch.Tensor,
    data_type: list | torch.Tensor = None,
    eps: float = 1e-6, # Unused in OT, but kept for signature compatibility
    token_level_scores_format: torch.Tensor = None, 
    token_level_scores_correctness: torch.Tensor = None, 
    token_level_scores_structure: torch.Tensor = None, 
    token_level_scores_length: torch.Tensor = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Task GRPO utilizing 1D Optimal Transport for Distributional Matching.
    Operates on the summed scalar outcome reward from token_level_rewards.
    """
    bsz = response_mask.shape[0]
    device = response_mask.device
    dtype = response_mask.dtype if torch.is_floating_point(response_mask) else torch.float32

    # --- 1. Base scores from rewards (Original GRPO logic) ---
    # Sum tokens to get a scalar outcome per response: (bs,)
    scores = token_level_rewards.sum(dim=-1).to(dtype)

    # --- 2. Helper to unify inputs to lists ---
    def _to_list(x):
        if x is None: return None
        if torch.is_tensor(x): return x.tolist()
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray): return x.tolist()
        except: pass
        return list(x) if isinstance(x, (list, tuple)) else [x]*bsz

    index_list = _to_list(index)
    pt_list = _to_list(problem_type)
    dt_list = _to_list(data_type)
    
    index_keys = [str(x) for x in index_list]
    
    # --- 3. Identify Tasks ---
    def _get_task_key(i):
        pt = str(pt_list[i]) if pt_list else "default"
        dt = str(dt_list[i]) if dt_list and dt_list[i] else ""
        if pt == "segmentation" and dt.lower() in ("video", "image"):
            return f"segmentation/{dt.lower()}"
        return pt

    task_to_indices = defaultdict(list)
    for i in range(bsz):
        task_to_indices[_get_task_key(i)].append(i)

    final_advantages_scalar = torch.zeros(bsz, device=device, dtype=dtype)

    # --- 4. Main Loop: Process Each Task Separately ---
    for task_key, batch_indices in task_to_indices.items():
        if not batch_indices: continue
        
        task_idxs = torch.tensor(batch_indices, device=device)
        task_scores = scores[task_idxs]
        task_group_ids = [index_keys[i] for i in batch_indices]

        # --- Intra-Prompt (Group) OT Normalization ---
        group_map = defaultdict(list)
        for local_i, gid in enumerate(task_group_ids):
            group_map[gid].append(local_i)
        
        ot_scores = torch.zeros_like(task_scores)
        
        for gid, local_indices in group_map.items():
            if len(local_indices) > 0:
                local_indices_t = torch.tensor(local_indices, device=device)
                # Apply OT mapping strictly within the prompt group
                ot_scores[local_indices_t] = compute_1d_ot(task_scores[local_indices_t])

        # --- Final Per-Task Whitening via OT ---
        if len(ot_scores) > 1:
            whitened_adv = compute_1d_ot(ot_scores)
        else:
            whitened_adv = torch.zeros_like(ot_scores)

        final_advantages_scalar[task_idxs] = whitened_adv

    # --- 5. Broadcast to Token Level and Return ---
    returns = final_advantages_scalar.unsqueeze(-1) * response_mask

    return returns, returns



@register_adv_estimator(AdvantageEstimator.GDPO)
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, 
    token_level_scores_format : torch.Tensor = None,
    token_level_scores_correctness : torch.Tensor = None,
    token_level_scores_structure : torch.Tensor = None, 
    token_level_scores_length : torch.Tensor = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    
    bsz = response_mask.shape[0]
    device = response_mask.device
    
    # 1. Define Helper for Intra-Group Normalization (GRPO Logic)
    def compute_group_normalized_score(token_scores_tensor):
        if token_scores_tensor is None:
            return torch.zeros(bsz, device=device)
            
        # Sum tokens to get scalar outcome per response: (bs,)
        scores = token_scores_tensor.sum(dim=-1)
        
        # Group scores by prompt ID (index)
        id2score = defaultdict(list)
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # Compute Mean/Std per group
        id2mean, id2std = {}, {}
        for idx, values in id2score.items():
            # GRPO requirement: Need >1 sample to calculate variance
            assert len(values) > 1, f"GDPO needs rollout.n > 1. Group {idx} has {len(values)} samples."
            val_tensor = torch.stack(values)
            id2mean[idx] = val_tensor.mean()
            id2std[idx] = val_tensor.std()

        # Normalize scores: (score - group_mean) / (group_std + eps)
        normalized = torch.zeros_like(scores)
        for i in range(bsz):
            idx = index[i]
            normalized[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + eps)
            
        return normalized

    # 2. Compute Independent Normalizations
    # We process each component separately as per GDPO paper/code
    norm_correctness = compute_group_normalized_score(token_level_scores_correctness)
    norm_format = compute_group_normalized_score(token_level_scores_format)
    norm_structure = compute_group_normalized_score(token_level_scores_structure)
    # norm_length = compute_group_normalized_score(token_level_scores_length)

    # 3. Combine Components
    total_advantage_scalar = norm_correctness + norm_format + norm_structure #+ norm_length 

    # Expand to token level for whitening and return
    # shape: (bs, response_length)
    advantages = total_advantage_scalar.unsqueeze(-1) * response_mask

    valid_token_count = response_mask.sum()
    
    # Global Mean
    global_mean = advantages.sum() / valid_token_count
    
    # Global Variance
    global_var = ((advantages - global_mean) ** 2 * response_mask).sum() / valid_token_count
    global_std = torch.sqrt(global_var)
    
    # Final Whitened Advantages
    final_advantages = (advantages - global_mean) / (global_std + eps)
    
    # Re-apply mask to ensure padding remains 0
    final_advantages = final_advantages * response_mask

    # For PPO/GRPO, returns often equal advantages when value estimation is implicit or handled this way
    return final_advantages, final_advantages


# --- 1D Optimal Transport Helper ---
def compute_1d_ot(rewards: torch.Tensor) -> torch.Tensor:
    """
    Maps a 1D tensor of rewards to a Standard Normal distribution N(0,1)
    using Optimal Transport (Inverse CDF mapping), handling ties.
    """
    N = rewards.shape[0]
    if N <= 1:
        return torch.zeros_like(rewards)
        
    sorted_rewards, indices = torch.sort(rewards)
    device = rewards.device
    dtype = rewards.dtype
    
    # Generate target quantiles from N(0,1)
    ranks = torch.arange(1, N + 1, device=device, dtype=torch.float32)
    probabilities = (ranks - 0.5) / N
    target_quantiles = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
    target_quantiles = target_quantiles.to(dtype)
    
    # Handle ties (identical rewards get the average of their target quantiles)
    unique_rewards, inverse_indices = torch.unique(sorted_rewards, return_inverse=True)
    if unique_rewards.shape[0] < N:
        target_quantiles_tied = torch.zeros_like(unique_rewards, dtype=dtype)
        target_quantiles_tied.scatter_add_(0, inverse_indices, target_quantiles)
        counts = torch.bincount(inverse_indices).to(dtype)
        target_quantiles_tied = target_quantiles_tied / counts
        target_quantiles = target_quantiles_tied[inverse_indices]

    advantages = torch.zeros_like(rewards)
    advantages[indices] = target_quantiles
    
    return advantages


@register_adv_estimator(AdvantageEstimator.GS_GDPO) 
def compute_pertask_gaussian_outcome_advantage_gdpo(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    index: list | torch.Tensor, 
    problem_type: list | torch.Tensor,
    data_type: list | torch.Tensor = None,
    eps: float = 1e-6, # Unused in OT, but kept for signature compatibility
    token_level_scores_format: torch.Tensor = None, 
    token_level_scores_correctness: torch.Tensor = None, 
    token_level_scores_structure: torch.Tensor = None, 
    token_level_scores_length: torch.Tensor = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Task GDPO utilizing 1D Optimal Transport for Distributional Matching.
    """
    bsz = response_mask.shape[0]
    device = response_mask.device
    dtype = response_mask.dtype if torch.is_floating_point(response_mask) else torch.float32

    # --- 1. Helper to unify inputs to lists ---
    def _to_list(x):
        if x is None: return None
        if torch.is_tensor(x): return x.tolist()
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray): return x.tolist()
        except: pass
        return list(x) if isinstance(x, (list, tuple)) else [x]*bsz

    index_list = _to_list(index)
    pt_list = _to_list(problem_type)
    dt_list = _to_list(data_type)
    
    index_keys = [str(x) for x in index_list]
    
    # --- 2. Identify Tasks ---
    def _get_task_key(i):
        pt = str(pt_list[i]) if pt_list else "default"
        dt = str(dt_list[i]) if dt_list and dt_list[i] else ""
        if pt == "segmentation" and dt.lower() in ("video", "image"):
            return f"segmentation/{dt.lower()}"
        return pt

    task_to_indices = defaultdict(list)
    for i in range(bsz):
        task_to_indices[_get_task_key(i)].append(i)

    final_advantages_scalar = torch.zeros(bsz, device=device, dtype=dtype)

    # --- 3. Main Loop: Process Each Task Separately ---
    for task_key, batch_indices in task_to_indices.items():
        if not batch_indices: continue
        
        task_idxs = torch.tensor(batch_indices, device=device)
        
        def get_component_slice(full_tensor):
            if full_tensor is None: 
                return torch.zeros(len(task_idxs), device=device, dtype=dtype)
            if full_tensor.dim() > 1:
                return full_tensor[task_idxs].sum(dim=-1).to(dtype)
            return full_tensor[task_idxs].to(dtype)

        comp_correctness = get_component_slice(token_level_scores_correctness)
        comp_format = get_component_slice(token_level_scores_format)
        comp_structure = get_component_slice(token_level_scores_structure)
        # comp_length = get_component_slice(token_level_scores_length)

        task_group_ids = [index_keys[i] for i in batch_indices]

        # --- Logic: Component OT Normalization (Intra-Prompt / Group Level) ---
        def normalize_component_ot(scores_slice):
            group_map = defaultdict(list)
            for local_i, gid in enumerate(task_group_ids):
                group_map[gid].append(local_i)
            
            ot_scores = torch.zeros_like(scores_slice)
            
            for gid, local_indices in group_map.items():
                if len(local_indices) > 0:
                    local_indices_t = torch.tensor(local_indices, device=device)
                    # Apply OT mapping strictly within the prompt group
                    ot_scores[local_indices_t] = compute_1d_ot(scores_slice[local_indices_t])
                    
            return ot_scores

        norm_correctness = normalize_component_ot(comp_correctness)
        norm_format = normalize_component_ot(comp_format)
        norm_structure = normalize_component_ot(comp_structure)
        # norm_length = normalize_component_ot(comp_length)

        # Sum Normalized Components
        task_total_adv = norm_correctness + norm_format + norm_structure #+ norm_length

        # --- 4. Final Per-Task Whitening via OT ---
        # Instead of (x - mu)/sigma, we map the summed task advantages 
        # to a perfect normal distribution across the whole task batch.
        if len(task_total_adv) > 1:
            whitened_adv = compute_1d_ot(task_total_adv)
        else:
            whitened_adv = torch.zeros_like(task_total_adv)

        final_advantages_scalar[task_idxs] = whitened_adv

    # --- 5. Broadcast to Token Level and Return ---
    advantages = final_advantages_scalar.unsqueeze(-1) * response_mask
    advantages = advantages * response_mask

    return advantages, advantages




# ===== NEW: task key generation function =====
def _task_key_of(sample_problem_type: str, sample_data_type: str | None) -> str:
    """
    Task partition rule:
    - By default, aggregate by problem_type;
    - If problem_type == "segmentation", further split by data_type into
      "segmentation/image" and "segmentation/video".
    """
    if sample_problem_type == "segmentation":
        dt = (sample_data_type or "").lower()
        if dt in ("video", "image"):
            return f"segmentation/{dt}"
    return sample_problem_type



@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    task_idx_mask_tensor_micro: torch.Tensor, 
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_avg_mode: Literal["token", "seq"],
    tau_positive: float,
    tau_negative: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo", "sapo"],
    use_entropy_loss: bool, 
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """

    negative_approx_kl = log_probs - old_log_probs
    # clamp negative_approx_kl to avoid nan kld
    negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}

    # use negative log probs as an estimator of entropy loss
    entropy_loss = average_loss(-log_probs, response_mask, mode=loss_avg_mode)
    metrics["entropy_loss"] =  entropy_loss

    pg_entropy_reg_loss =  entropy_loss * 0.0 
    if use_entropy_loss:
        task_type_to_id = {
            'SG': 0, 
            'SR': 1, 
            "GVQA":2, 
            'Math':3, 
            'Chart': 4, 
            'OCR':5, 
        }
        for task_type in ['SG', "SR" , 'GVQA', 'Math', 'Chart', 'OCR']: 
            task_mask = task_idx_mask_tensor_micro == task_type_to_id[task_type]
            task_log_probs = log_probs[task_mask, ...] 
            task_response_mask = response_mask[task_mask, ...] 
            task_entropy = average_loss(-task_log_probs, task_response_mask, mode=loss_avg_mode)
            metrics[f"entropy_loss_{task_type}"] = task_entropy

            if use_entropy_loss:
                high = 0.6  #0.4  # 0.5
                if task_entropy > high: 
                    reg = torch.relu(task_entropy - high)   
                elif task_entropy < 0.2:           
                    reg = torch.relu(0.2 - task_entropy)
                else: 
                    reg = task_entropy * 0.0 
                pg_entropy_reg_loss += reg

    if loss_type == "cispo":

        final_pg_loss = -advantages * log_probs * clipped_ratio.detach()

    elif loss_type == "sapo":

        positive_token_mask =  (advantages >= 0).float()
        negative_token_mask =  (advantages < 0).float()

        gate_negative = 4.0 / tau_negative * torch.sigmoid(tau_negative * (ratio - 1.0))
        gate_positive = 4.0 / tau_positive * torch.sigmoid(tau_positive * (ratio - 1.0))

        final_pg_loss = -advantages * (positive_token_mask * gate_positive + negative_token_mask * gate_negative)
    else:
        pg_loss = -advantages * ratio  # -ratio * A
        pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
        pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
        metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}

    return final_pg_loss, metrics , pg_entropy_reg_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
        vpred_mean: a float
            The mean of predicted values

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
