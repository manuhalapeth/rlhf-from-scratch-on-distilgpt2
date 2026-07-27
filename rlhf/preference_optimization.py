"""Direct preference optimization and its relatives.

DPO, IPO, KTO, ORPO, and SimPO all replace PPO's explicit reward model
and rollout loop with a closed form loss computed directly on chosen and
rejected log probabilities. They differ in what target they push the
preference margin toward and whether they need a frozen reference model
at all. See docs/TRADEOFFS.md for a full comparison.
"""
import torch
import torch.nn.functional as F


def dpo_logratios(chosen_logps, rejected_logps):
    """Per pair difference between chosen and rejected log probabilities under the policy."""
    return chosen_logps - rejected_logps


def dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps):
    """Per pair difference between chosen and rejected log probabilities under the frozen reference model."""
    return ref_chosen_logps - ref_rejected_logps


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Direct Preference Optimization loss from policy and reference log probabilities."""
    policy_logratios = dpo_logratios(policy_chosen_logps, policy_rejected_logps)
    ref_logratios = dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps)
    logits = beta * (policy_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def ipo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Identity Preference Optimization loss: squared error between the log ratio gap and the IPO target margin."""
    policy_logratios = dpo_logratios(policy_chosen_logps, policy_rejected_logps)
    ref_logratios = dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps)
    gap = policy_logratios - ref_logratios
    target = 1.0 / (2.0 * beta)
    return ((gap - target) ** 2).mean()


def kto_loss(policy_logps, ref_logps, labels, beta=0.1):
    """Kahneman Tversky Optimization loss for unpaired desirable and undesirable examples."""
    policy_logps = torch.as_tensor(policy_logps, dtype=torch.float32)
    ref_logps = torch.as_tensor(ref_logps, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)

    logratios = policy_logps - ref_logps
    signed_logits = beta * logratios * (2 * labels - 1)
    return (1 - torch.sigmoid(signed_logits)).mean()


def _log1mexp(x):
    """Numerically stable log(1 minus exp(x)) for x less than or equal to zero."""
    return torch.where(
        x > -0.693147,  # log(0.5)
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def _log_odds(logp):
    """log(p over (1 minus p)) computed stably from log(p)."""
    return logp - _log1mexp(logp)


def orpo_loss(policy_chosen_logps, policy_rejected_logps, sft_loss, lambda_or=0.1):
    """ORPO loss: supervised finetuning loss plus a lambda weighted odds ratio preference penalty."""
    chosen_log_odds = _log_odds(policy_chosen_logps)
    rejected_log_odds = _log_odds(policy_rejected_logps)
    delta = chosen_log_odds - rejected_log_odds
    or_loss = -F.logsigmoid(delta).mean()
    return sft_loss + lambda_or * or_loss


def simpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    chosen_lengths,
    rejected_lengths,
    beta=2.0,
    gamma=0.5,
):
    """SimPO loss: reference free preference loss using length normalized log probabilities as implicit rewards."""
    chosen_reward = policy_chosen_logps / chosen_lengths
    rejected_reward = policy_rejected_logps / rejected_lengths
    logits = beta * (chosen_reward - rejected_reward) - gamma
    return -F.logsigmoid(logits).mean()
