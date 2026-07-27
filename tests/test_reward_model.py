"""Unit tests for the reward model math that does not require a backbone forward pass."""
import torch

from rlhf.reward_model import pairwise_accuracy, pairwise_reward_loss, reward_bce_loss, reward_head_forward


def test_reward_head_forward_is_a_dot_product_plus_bias():
    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    weight = torch.tensor([[0.5, 0.5, 0.5]])  # shape (1, hidden) like nn.Linear(hidden, 1).weight
    bias = torch.tensor([1.0])
    reward = reward_head_forward(hidden, weight, bias)
    assert torch.isclose(reward, torch.tensor([4.0]))  # (1+2+3) * 0.5 + 1


def test_pairwise_reward_loss_decreases_as_the_margin_grows():
    small = pairwise_reward_loss(torch.tensor([1.0]), torch.tensor([0.5]))
    large = pairwise_reward_loss(torch.tensor([2.0]), torch.tensor([0.5]))
    assert large < small


def test_pairwise_reward_loss_is_symmetric_around_chance_at_zero_margin():
    equal = pairwise_reward_loss(torch.tensor([0.0]), torch.tensor([0.0]))
    assert torch.isclose(equal, torch.tensor(0.6931), atol=1e-3)  # log(2)


def test_reward_bce_loss_rewards_correct_separation():
    correct = reward_bce_loss(torch.tensor([3.0]), torch.tensor([-3.0]))
    incorrect = reward_bce_loss(torch.tensor([-3.0]), torch.tensor([3.0]))
    assert correct < incorrect


def test_pairwise_accuracy_counts_correctly_ordered_pairs():
    chosen = torch.tensor([1.0, 0.0, 5.0])
    rejected = torch.tensor([0.0, 1.0, 4.0])
    # pair 0: correct, pair 1: wrong, pair 2: correct -> 2/3
    assert abs(pairwise_accuracy(chosen, rejected) - (2 / 3)) < 1e-6
