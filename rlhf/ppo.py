"""Proximal Policy Optimization primitives for reinforcement learning from human feedback.

Sequence log probabilities, a KL estimate against a frozen reference
policy, discounted returns, generalized advantage estimation, the
clipped surrogate objective, a value function loss, an entropy bonus,
and the combined PPO loss that ties them together.
"""
import numpy as np
import torch
import torch.nn.functional as F


def sequence_logprob(logits, token_ids):
    """Sum log probabilities of the selected tokens along the sequence dimension."""
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


def per_token_kl(policy_logprobs, ref_logprobs):
    """Per token KL estimate between policy and reference log probabilities."""
    return policy_logprobs - ref_logprobs


def compute_returns(rewards, gamma=0.99):
    """Return the discounted return at each timestep as a one dimensional numpy array."""
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards, dtype=np.float64)

    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def gae_advantages(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation, returning advantages of shape (T,)."""
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)

    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


def policy_ratio(new_logprobs, old_logprobs):
    """PPO importance sampling ratio: exponential of (new log probability minus old log probability)."""
    return torch.exp(new_logprobs - old_logprobs)


def clipped_surrogate(ratio, advantages, clip_eps=0.2):
    """PPO clipped surrogate loss (to minimize) from ratio and advantages."""
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    objective = torch.min(unclipped, clipped)
    return -objective.mean()


def value_function_loss(values, returns):
    """Mean squared error between predicted values and computed returns."""
    return ((values - returns) ** 2).mean()


def entropy_bonus(logits):
    """Mean per position categorical entropy over the vocabulary axis."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    per_position_entropy = -(probs * log_probs).sum(dim=-1)
    return per_position_entropy.mean()


def ppo_loss(ratio, advantages, values, returns, logits, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """Combine PPO's clipped policy surrogate, value loss, and entropy bonus into the total loss."""
    policy_loss = clipped_surrogate(ratio, advantages, clip_eps=clip_eps)
    value_loss = value_function_loss(values, returns)
    entropy = entropy_bonus(logits)
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "loss": total_loss,
    }


def kl_penalized_reward(reward, kl, beta=0.1):
    """Combine reward model score with a KL penalty against the reference policy."""
    return reward - beta * kl


def batch_sequence_logprob(logits, token_ids, attention_mask=None):
    """Batched version of sequence_logprob: sum of token log probabilities per sequence, shape (B,)."""
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

    if attention_mask is not None:
        token_log_probs = token_log_probs * attention_mask.to(token_log_probs.dtype)

    return token_log_probs.sum(dim=-1)
