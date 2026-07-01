"""Entropic Adaptive Beta advantage estimator for GRPO-style group RL.

Ported from TTT-Discover (ttt_discover/rl/train.py). For each group of
completions sharing the same prompt, finds beta via binary search such
that KL(q_beta || uniform) = log(2), then computes leave-one-out (LOO)
Boltzmann weights as advantages.
"""

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est

try:
    from verl.utils.config import AlgoConfig
except ImportError:
    AlgoConfig = None


def _solve_beta_for_group(rewards: torch.Tensor, delta: float = math.log(2),
                          beta_max: float = 1e6, iters: int = 60) -> torch.Tensor:
    """Binary search for beta where KL(q_beta || uniform) = delta.

    Args:
        rewards: 1-D tensor of per-sequence rewards within one group.
        delta: target KL divergence (default log(2)).
        beta_max: upper bound for binary search.
        iters: number of binary search iterations.

    Returns:
        Scalar tensor beta.
    """
    r = rewards.float()
    k = r.shape[0]

    if k < 2:
        return r.new_tensor(0.0)

    log_k = math.log(k)

    def kl_hat(beta_scalar: float) -> float:
        b = r.new_tensor(beta_scalar)
        logits = b * (r - r.max(dim=0, keepdim=True).values)
        logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        q = torch.exp(logq)
        kl = (q * (logq + log_k)).sum(dim=0)
        return float(kl.mean().item())

    lo, hi = 0.0, 1.0
    if kl_hat(hi) < delta:
        while hi < beta_max and kl_hat(hi) < delta:
            hi *= 2.0
        if kl_hat(hi) < delta:
            return r.new_tensor(hi)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kl_hat(mid) < delta:
            lo = mid
        else:
            hi = mid

    return r.new_tensor(hi)


def _entropic_advantages(rewards: torch.Tensor, beta: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    """Compute LOO Boltzmann advantages given beta.

    Returns per-sequence advantages (w - 1) where w = exp(beta*r) / Z_loo.
    """
    k = rewards.shape[0]
    e = torch.exp(beta * (rewards - rewards.max(dim=0, keepdim=True).values))

    if k == 1:
        z_loo = e
    else:
        z_loo = (e.sum(dim=0, keepdim=True) - e) / (k - 1)

    w = e / (z_loo + eps)
    return w - 1.0


@register_adv_est("entropic_adaptive_beta")
def compute_entropic_adaptive_beta_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional["AlgoConfig"] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute entropic adaptive beta advantages grouped by prompt index.

    For each group of completions sharing the same prompt:
    1. Aggregate token-level rewards to per-sequence scalar.
    2. Binary-search for beta where KL(q_beta || uniform) = log(2).
    3. Compute LOO Boltzmann weights: advantages = w - 1.
    4. Broadcast to token level via response_mask.

    Args:
        token_level_rewards: shape (batch, response_length).
        response_mask: shape (batch, response_length), 1 for response tokens.
        index: shape (batch,), group membership identifier.
        epsilon: unused, kept for interface compatibility.
        config: optional algorithm config.

    Returns:
        (advantages, returns): both shape (batch, response_length).
    """
    scores = token_level_rewards.sum(dim=-1)  # (batch,)

    id2indices = defaultdict(list)
    bsz = scores.shape[0]
    for i in range(bsz):
        id2indices[index[i]].append(i)

    per_seq_advantages = torch.zeros_like(scores)

    with torch.no_grad():
        for idx, members in id2indices.items():
            group_rewards = scores[members]
            # Skip groups where all rewards are identical (no gradient signal)
            if len(members) > 1 and group_rewards.max() == group_rewards.min():
                continue  # per_seq_advantages stays 0 for this group
            beta = _solve_beta_for_group(group_rewards)
            group_adv = _entropic_advantages(group_rewards, beta)
            for j, member_idx in enumerate(members):
                per_seq_advantages[member_idx] = group_adv[j]

    token_advantages = per_seq_advantages.unsqueeze(-1) * response_mask

    # Apply centered KL penalty adjustment (matches original TTT-Discover)
    old_log_probs = kwargs.get("old_log_probs")
    ref_log_prob = kwargs.get("ref_log_prob")
    if old_log_probs is not None and ref_log_prob is not None:
        kl_coef = getattr(config, "kl_ctrl", None)
        kl_coef = getattr(kl_coef, "kl_coef", 0.1) if kl_coef else 0.1
        with torch.no_grad():
            diff = (old_log_probs - ref_log_prob) * response_mask
            mask_sum = response_mask.sum()
            avg_diff = diff.sum() / mask_sum.clamp(min=1.0)
            kl_adjustment = kl_coef * response_mask * (avg_diff - diff)
            token_advantages = token_advantages + kl_adjustment

    return token_advantages, token_advantages
