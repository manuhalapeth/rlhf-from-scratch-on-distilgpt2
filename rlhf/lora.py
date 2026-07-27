"""Low rank adaptation: parameter efficient finetuning primitives.

A LoRA layer freezes a base weight matrix and learns a low rank update
around it, so most of the model stays fixed while a tiny adapter carries
the new gradient signal. merge_lora folds the adapter back into the base
weight for adapter free inference once training is done.
"""
import torch
import torch.nn.functional as F


def lora_delta(A, B, alpha, r):
    """Build the scaled low rank weight update: (alpha / r) times B times A."""
    scaling = alpha / r
    return scaling * (B @ A)


def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    """Run a LoRA augmented linear layer: x times (base_weight plus the low rank delta), transposed, plus bias."""
    effective_weight = base_weight + lora_delta(A, B, alpha, r)
    return F.linear(x, effective_weight, bias)


def init_lora_weights(in_features, out_features, r, seed=0):
    """Create LoRA factor tensors A (small random) and B (zeros), so the adapter starts as a no op."""
    torch.manual_seed(seed)
    A = torch.randn(r, in_features, dtype=torch.float32) * 0.01
    B = torch.zeros(out_features, r, dtype=torch.float32)
    return A, B


def freeze_base_params(model):
    """Disable gradients on all non LoRA parameters, leaving LoRA params trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
    return model


def count_trainable_params(model):
    """Sum the element counts of all parameters currently marked trainable."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def merge_lora(base_weight, lora_a, lora_b, scaling):
    """Fold a trained LoRA update into the base weight for adapter free inference."""
    return base_weight + scaling * (lora_b @ lora_a)
