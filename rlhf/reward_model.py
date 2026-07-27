"""Reward modeling: turn a backbone's hidden states into a scalar preference score.

The reward model is a frozen language model backbone plus a trained
linear head, fit with a Bradley Terry pairwise loss so that
reward(chosen) ends up higher than reward(rejected).
"""
import torch
import torch.nn.functional as F


def reward_head_forward(hidden_state, weight, bias):
    """Project final hidden states into a scalar reward per example."""
    weight = weight.reshape(-1)
    return hidden_state @ weight + bias


def pairwise_reward_loss(chosen_rewards, rejected_rewards):
    """Bradley Terry pairwise loss: mean of negative log sigmoid of (chosen minus rejected)."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def reward_bce_loss(chosen_reward, rejected_reward):
    """Binary cross entropy style reward loss: chosen pushed toward label one, rejected toward label zero."""
    chosen_reward = torch.as_tensor(chosen_reward, dtype=torch.float32)
    rejected_reward = torch.as_tensor(rejected_reward, dtype=torch.float32)

    chosen_loss = F.binary_cross_entropy_with_logits(
        chosen_reward, torch.ones_like(chosen_reward)
    )
    rejected_loss = F.binary_cross_entropy_with_logits(
        rejected_reward, torch.zeros_like(rejected_reward)
    )
    return (chosen_loss + rejected_loss) / 2


def pairwise_accuracy(chosen_reward, rejected_reward):
    """Fraction of pairs where chosen_reward is greater than rejected_reward."""
    return (chosen_reward > rejected_reward).float().mean().item()


def _last_token_hidden(model, input_ids, attention_mask):
    """Run the backbone and gather each sequence's last non pad hidden state."""
    hidden_states = model(input_ids, attention_mask=attention_mask)
    if hasattr(hidden_states, "last_hidden_state"):
        hidden_states = hidden_states.last_hidden_state

    last_idx = attention_mask.long().sum(dim=1) - 1
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, last_idx]


def reward_train_step(model, reward_head, batch, optimizer):
    """Run one optimization step of the reward model on a paired preference batch."""
    optimizer.zero_grad()

    chosen_hidden = _last_token_hidden(
        model, batch["chosen_input_ids"], batch["chosen_attention_mask"]
    )
    rejected_hidden = _last_token_hidden(
        model, batch["rejected_input_ids"], batch["rejected_attention_mask"]
    )

    chosen_reward = reward_head_forward(chosen_hidden, reward_head.weight, reward_head.bias)
    rejected_reward = reward_head_forward(rejected_hidden, reward_head.weight, reward_head.bias)

    loss = pairwise_reward_loss(chosen_reward, rejected_reward)
    accuracy = pairwise_accuracy(chosen_reward, rejected_reward)

    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "accuracy": accuracy}


def score_with_reward(reward_model, tokenizer, prompt, completion):
    """Score a prompt and completion pair using the reward model's backbone plus reward head."""
    model = reward_model["model"]
    weight = reward_model["weight"]
    bias = reward_model["bias"]

    text = prompt + completion
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_hidden = hidden_states[:, -1, :]
        reward = reward_head_forward(last_hidden, weight, bias)

    return reward.item()


def win_rate(reward_model, tokenizer, prompts, completions_a, completions_b):
    """Fraction of prompts where completions_a beats completions_b under the reward model, ties count as half."""
    total_score = 0.0
    n = len(prompts)

    for i in range(n):
        score_a = score_with_reward(reward_model, tokenizer, prompts[i], completions_a[i])
        score_b = score_with_reward(reward_model, tokenizer, prompts[i], completions_b[i])

        if score_a > score_b:
            total_score += 1.0
        elif score_a == score_b:
            total_score += 0.5

    return total_score / n
