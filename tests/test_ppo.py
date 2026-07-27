"""Unit tests for the PPO building blocks: advantages, ratios, and the clipped surrogate."""
import numpy as np
import torch

from rlhf.ppo import (
    clipped_surrogate,
    compute_returns,
    entropy_bonus,
    gae_advantages,
    kl_penalized_reward,
    per_token_kl,
    policy_ratio,
    ppo_loss,
    value_function_loss,
)


def test_compute_returns_matches_hand_computed_discounting():
    rewards = [1.0, 1.0, 1.0]
    returns = compute_returns(rewards, gamma=0.5)
    expected = np.array([1 + 0.5 * (1 + 0.5 * 1), 1 + 0.5 * 1, 1.0])
    assert np.allclose(returns, expected)


def test_policy_ratio_is_one_when_new_equals_old():
    logp = torch.tensor([-1.0, -2.0])
    ratio = policy_ratio(logp, logp)
    assert torch.allclose(ratio, torch.ones_like(logp))


def test_clipped_surrogate_clips_large_positive_advantage_ratio():
    ratio = torch.tensor([2.0])  # far above 1 + clip_eps
    advantages = torch.tensor([1.0])
    loss = clipped_surrogate(ratio, advantages, clip_eps=0.2)
    # clipped objective caps the ratio at 1.2, so loss should equal -1.2
    assert torch.isclose(loss, torch.tensor(-1.2))


def test_clipped_surrogate_does_not_clip_within_range():
    ratio = torch.tensor([1.05])
    advantages = torch.tensor([1.0])
    loss = clipped_surrogate(ratio, advantages, clip_eps=0.2)
    assert torch.isclose(loss, torch.tensor(-1.05))


def test_per_token_kl_is_zero_when_policy_matches_reference():
    logp = torch.tensor([-1.0, -2.0, -3.0])
    kl = per_token_kl(logp, logp)
    assert torch.equal(kl, torch.zeros_like(logp))


def test_kl_penalized_reward_subtracts_the_penalty():
    result = kl_penalized_reward(reward=2.0, kl=1.0, beta=0.5)
    assert result == 1.5


def test_gae_advantages_reduces_to_one_step_td_error_when_lambda_is_zero():
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.0, 0.5, 0.0])  # includes the bootstrap value
    advantages = gae_advantages(rewards, values, gamma=1.0, lam=0.0)
    expected_t1 = rewards[1] + 1.0 * values[2] - values[1]
    expected_t0 = rewards[0] + 1.0 * values[1] - values[0]
    assert torch.isclose(advantages[1], expected_t1)
    assert torch.isclose(advantages[0], expected_t0)


def test_value_function_loss_is_zero_for_a_perfect_predictor():
    values = torch.tensor([1.0, 2.0, 3.0])
    assert value_function_loss(values, values) == 0.0


def test_entropy_bonus_is_higher_for_a_uniform_distribution():
    uniform_logits = torch.zeros(1, 5)
    peaked_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    assert entropy_bonus(uniform_logits) > entropy_bonus(peaked_logits)


def test_ppo_loss_combines_all_three_terms_additively():
    ratio = torch.tensor([1.0])
    advantages = torch.tensor([1.0])
    values = torch.tensor([0.0])
    returns = torch.tensor([1.0])
    logits = torch.zeros(1, 4)
    out = ppo_loss(ratio, advantages, values, returns, logits, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    expected_total = out["policy_loss"] + 0.5 * out["value_loss"] - 0.01 * out["entropy"]
    assert torch.isclose(out["loss"], expected_total)
