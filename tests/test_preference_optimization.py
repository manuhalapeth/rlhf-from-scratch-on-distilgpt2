"""Unit tests for DPO, IPO, KTO, ORPO, and SimPO.

Rather than asserting exact numbers, these check the property that
actually matters: the loss should get smaller as the policy separates
chosen from rejected further in the preferred direction, and should not
reward moving in the wrong direction. That property is what "the loss
is correctly implemented" means for a preference optimization method.
"""
import torch

from rlhf.preference_optimization import dpo_loss, ipo_loss, kto_loss, orpo_loss, simpo_loss


def test_dpo_loss_prefers_a_larger_positive_margin():
    ref_chosen = torch.tensor([-5.0])
    ref_rejected = torch.tensor([-5.0])
    small_margin = dpo_loss(torch.tensor([-4.0]), torch.tensor([-5.0]), ref_chosen, ref_rejected, beta=0.1)
    large_margin = dpo_loss(torch.tensor([-2.0]), torch.tensor([-5.0]), ref_chosen, ref_rejected, beta=0.1)
    assert large_margin < small_margin


def test_dpo_loss_is_worse_when_policy_prefers_rejected():
    ref_chosen = torch.tensor([-5.0])
    ref_rejected = torch.tensor([-5.0])
    correct_direction = dpo_loss(torch.tensor([-4.0]), torch.tensor([-6.0]), ref_chosen, ref_rejected, beta=0.1)
    wrong_direction = dpo_loss(torch.tensor([-6.0]), torch.tensor([-4.0]), ref_chosen, ref_rejected, beta=0.1)
    assert correct_direction < wrong_direction


def test_ipo_loss_is_minimized_at_its_target_margin_not_beyond_it():
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    beta = 0.5
    target = 1.0 / (2.0 * beta)  # ipo_loss's implicit target gap

    at_target = ipo_loss(torch.tensor([target]), torch.tensor([0.0]), ref_chosen, ref_rejected, beta=beta)
    overshot = ipo_loss(torch.tensor([target * 3]), torch.tensor([0.0]), ref_chosen, ref_rejected, beta=beta)
    assert at_target < overshot  # unlike DPO, pushing the margin further is penalized


def test_kto_loss_rewards_desirable_examples_moving_above_reference():
    policy = torch.tensor([-1.0, -5.0])
    ref = torch.tensor([-3.0, -3.0])
    labels = torch.tensor([1.0, 0.0])  # first is desirable, second is undesirable
    loss = kto_loss(policy, ref, labels, beta=0.1)
    worse_policy = kto_loss(torch.tensor([-5.0, -1.0]), ref, labels, beta=0.1)
    assert loss < worse_policy


def test_orpo_loss_adds_a_nonnegative_odds_ratio_penalty_to_sft_loss():
    sft_loss = torch.tensor(1.5)
    chosen = torch.tensor([-0.2])  # length normalized log prob, closer to 0 means more likely
    rejected = torch.tensor([-2.0])
    loss = orpo_loss(chosen, rejected, sft_loss, lambda_or=0.1)
    assert loss > sft_loss  # the odds ratio term is strictly positive here


def test_simpo_loss_is_reference_free_and_rewards_length_normalized_margin():
    chosen_logps = torch.tensor([-2.0])
    rejected_logps = torch.tensor([-6.0])
    lengths = torch.tensor([4.0])
    good = simpo_loss(chosen_logps, rejected_logps, lengths, lengths, beta=2.0, gamma=0.5)
    flipped = simpo_loss(rejected_logps, chosen_logps, lengths, lengths, beta=2.0, gamma=0.5)
    assert good < flipped
