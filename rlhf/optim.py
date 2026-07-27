"""From scratch optimizer and schedule primitives.

A hand rolled AdamW step, linear warmup schedule, gradient norm clipping,
and micro batch gradient averaging, implemented directly against tensors
rather than delegated to torch.optim. Used to train the LoRA adapter in
run_pipeline.py so at least one training loop in this project does not
depend on a library optimizer.
"""
import torch


def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in place AdamW step to param using grad and persistent state."""
    if "step" not in state:
        state["step"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)

    beta1, beta2 = betas
    state["step"] += 1
    step = state["step"]

    m = state["m"]
    v = state["v"]

    if weight_decay != 0:
        param.data.mul_(1 - lr * weight_decay)

    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    denom = (v / bias_correction2).sqrt().add_(eps)
    step_size = lr / bias_correction1

    param.data.addcdiv_(m, denom, value=-step_size)
    return state


def linear_warmup_schedule(step, warmup_steps):
    """Return a linear warmup multiplier between zero and one for the current step."""
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return step / warmup_steps


def clip_grad_norm(grads, max_norm):
    """Compute the global L2 norm of grads and rescale in place if it exceeds max_norm."""
    total_norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    total_norm_value = float(total_norm)

    if total_norm_value > max_norm:
        scale = max_norm / (total_norm_value + 1e-6)
        for g in grads:
            g.detach().mul_(scale)

    return total_norm_value


def accumulate_gradients(grad_list):
    """Average a list of equally shaped gradient tensors across micro batches."""
    return torch.stack(grad_list, dim=0).mean(dim=0)
