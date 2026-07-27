"""Unit tests for the from scratch optimizer and schedule primitives."""
import torch

from rlhf.optim import accumulate_gradients, adamw_update, clip_grad_norm, linear_warmup_schedule


def test_linear_warmup_schedule_ramps_from_zero_to_one():
    assert linear_warmup_schedule(0, warmup_steps=10) == 0.0
    assert linear_warmup_schedule(5, warmup_steps=10) == 0.5
    assert linear_warmup_schedule(10, warmup_steps=10) == 1.0
    assert linear_warmup_schedule(20, warmup_steps=10) == 1.0


def test_linear_warmup_schedule_disabled_returns_one():
    assert linear_warmup_schedule(0, warmup_steps=0) == 1.0


def test_clip_grad_norm_rescales_when_over_budget():
    grads = [torch.tensor([3.0, 4.0])]  # norm is 5
    reported_norm = clip_grad_norm(grads, max_norm=1.0)
    assert reported_norm == 5.0
    new_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    assert torch.isclose(new_norm, torch.tensor(1.0), atol=1e-4)


def test_clip_grad_norm_leaves_small_grads_untouched():
    grads = [torch.tensor([0.1, 0.2])]
    before = grads[0].clone()
    clip_grad_norm(grads, max_norm=10.0)
    assert torch.equal(grads[0], before)


def test_accumulate_gradients_averages_micro_batches():
    grads = [torch.tensor([1.0, 1.0]), torch.tensor([3.0, 5.0])]
    averaged = accumulate_gradients(grads)
    assert torch.equal(averaged, torch.tensor([2.0, 3.0]))


def test_adamw_update_reduces_a_toy_quadratic_loss():
    # Minimize f(x) = (x - 3)^2 by hand rolled gradient descent through adamw_update.
    param = torch.tensor([0.0], requires_grad=False)
    state = {}
    for _ in range(200):
        grad = 2 * (param - 3.0)
        adamw_update(param, grad, state, lr=0.05)
    assert torch.isclose(param, torch.tensor([3.0]), atol=0.05)
