"""Supervised finetuning: the next token cross entropy training loop."""
import torch
import torch.nn.functional as F


def shift_logits_and_labels(logits, labels):
    """Drop the last logit position and the first label position so
    predictions at time t align with labels at time t+1."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return shift_logits, shift_labels


def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next token cross entropy, ignoring label positions equal to the ignore index."""
    vocab_size = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def sft_train_step(model, batch, optimizer):
    """Run one supervised finetuning forward, backward, and optimizer step; return the loss as a float."""
    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits

    shift_logits, shift_labels = shift_logits_and_labels(logits, batch["labels"])
    loss = cross_entropy_loss(shift_logits, shift_labels)

    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_loss(model, batches):
    """Mean language modeling loss over validation batches, no gradient tracking."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in batches:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits

            shift_logits, shift_labels = shift_logits_and_labels(logits, batch["labels"])
            loss = cross_entropy_loss(shift_logits, shift_labels)

            total_loss += loss.item()
            num_batches += 1

    if was_training:
        model.train()

    return total_loss / num_batches
