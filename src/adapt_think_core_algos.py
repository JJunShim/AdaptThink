import math

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


def masked_mean(values, mask, dim=-1, eps=1e-8):
    mask_count = mask.sum(dim=dim) + eps
    return (values * mask).sum(dim=dim) / mask_count


def masked_std(values, mean, mask, dim=-1, eps=1e-8):
    mask_count = mask.sum(dim=dim) + eps
    masked_var = ((values - mean) ** 2 * mask).sum(dim=dim) / mask_count
    return torch.sqrt(masked_var + eps)


def whiten(x, mask=None, dim=-1, eps=1e-8):
    if mask is None:
        # Standard normalization without mask
        mean = x.mean(dim=dim)
        std = x.std(dim=dim, unbiased=False)
    else:
        # Masked normalization
        # Convert mask to same dtype as x for proper computation
        mask = mask.to(x.dtype)

        mean = masked_mean(x, mask, dim, eps)
        std = masked_std(x, mean, mask, dim, eps)

    # Normalize
    return (x - mean) / (std + eps)


def compute_batchwise_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    dim = -1
    scores = token_level_rewards.sum(dim=dim)
    # scores = verl_F.masked_mean(token_level_rewards, response_mask, axis=dim)
    # scores = masked_mean(token_level_rewards, response_mask, eps=epsilon)

    with torch.no_grad():
        scores = whiten(scores, dim=dim, eps=epsilon)
        advantages = scores.unsqueeze(dim) * response_mask

    return advantages, advantages


def compute_naive_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    scores = token_level_rewards.sum(dim=-1)
    with torch.no_grad():
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(
            loss_mask, dim=-1
        )  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
    adapt_think_adjust_old_log_prob=False,
    nothinking_ratio=0,
    enforce_nothinking=None,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert (
        clip_ratio_c > 1.0
    ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    if adapt_think_adjust_old_log_prob:
        # print(f"\n\nOLD LOG PROBS\n{old_log_prob[:, 0].exp()}")
        old_log_prob[enforce_nothinking, 0] = (
            math.log(nothinking_ratio) if nothinking_ratio > 0 else -1e9
        )
        old_log_prob[~enforce_nothinking, 0] = (
            math.log(1 - nothinking_ratio) if nothinking_ratio < 1 else -1e9
        )

        log_prob = verl_F.masked_mean(log_prob, response_mask, axis=-1)  # (bs, )
        old_log_prob = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
        advantages = verl_F.masked_mean(advantages, response_mask, axis=-1)

        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)

        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        pg_losses = torch.maximum(pg_losses1, pg_losses2)
        pg_loss = torch.mean(pg_losses)

        return pg_loss, None, None, None
    else:
        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        pg_losses1 = -advantages * ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(
            pg_losses1, pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = verl_F.masked_mean(
            torch.gt(pg_losses2, pg_losses1).float(), response_mask
        )

        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(),
            response_mask,
        )

        # pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        pg_losses = clip_pg_losses1
        pg_loss = agg_loss(
            loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
        )

        return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
