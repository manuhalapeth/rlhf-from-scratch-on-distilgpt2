"""Unit tests for the LoRA adapter math."""
import torch

from rlhf.lora import (
    count_trainable_params,
    freeze_base_params,
    init_lora_weights,
    lora_delta,
    lora_linear_forward,
    merge_lora,
)


def test_init_lora_weights_shapes_and_zero_b():
    A, B = init_lora_weights(in_features=8, out_features=4, r=2, seed=0)
    assert A.shape == (2, 8)
    assert B.shape == (4, 2)
    assert torch.equal(B, torch.zeros_like(B))


def test_lora_delta_matches_scaled_matmul():
    A = torch.randn(2, 8)
    B = torch.randn(4, 2)
    delta = lora_delta(A, B, alpha=16, r=2)
    assert torch.allclose(delta, 8.0 * (B @ A))


def test_lora_linear_forward_is_a_no_op_when_b_is_zero():
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    base_weight = torch.randn(4, 8)
    A, B = init_lora_weights(in_features=8, out_features=4, r=2, seed=0)
    out = lora_linear_forward(x, base_weight, A, B, alpha=16, r=2)
    expected = torch.nn.functional.linear(x, base_weight)
    assert torch.allclose(out, expected)


def test_lora_linear_forward_moves_away_from_base_once_b_is_nonzero():
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    base_weight = torch.randn(4, 8)
    A, B = init_lora_weights(in_features=8, out_features=4, r=2, seed=0)
    B = B + 0.5  # simulate a trained, nonzero adapter
    out = lora_linear_forward(x, base_weight, A, B, alpha=16, r=2)
    base_out = torch.nn.functional.linear(x, base_weight)
    assert not torch.allclose(out, base_out)


def test_merge_lora_matches_lora_linear_forward():
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    base_weight = torch.randn(4, 8)
    A, B = init_lora_weights(in_features=8, out_features=4, r=2, seed=0)
    B = B + 0.3
    scaling = 16 / 2
    merged_weight = merge_lora(base_weight, A, B, scaling)
    merged_out = torch.nn.functional.linear(x, merged_weight)
    adapter_out = lora_linear_forward(x, base_weight, A, B, alpha=16, r=2)
    assert torch.allclose(merged_out, adapter_out, atol=1e-6)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4)
        self.lora_a = torch.nn.Parameter(torch.randn(2, 4))
        self.lora_b = torch.nn.Parameter(torch.zeros(4, 2))


def test_freeze_base_params_only_leaves_lora_named_params_trainable():
    model = _TinyModel()
    freeze_base_params(model)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable_names == {"lora_a", "lora_b"}


def test_count_trainable_params_after_freeze():
    model = _TinyModel()
    freeze_base_params(model)
    assert count_trainable_params(model) == model.lora_a.numel() + model.lora_b.numel()
