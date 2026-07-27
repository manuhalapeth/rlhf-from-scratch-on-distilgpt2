"""Unit tests for the shift and cross entropy pieces of the supervised finetuning loss."""
import torch

from rlhf.sft import cross_entropy_loss, shift_logits_and_labels


def test_shift_logits_and_labels_drops_the_correct_ends():
    logits = torch.zeros(1, 5, 3)
    labels = torch.tensor([[10, 11, 12, 13, 14]])
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    assert shift_logits.shape == (1, 4, 3)
    assert shift_labels.tolist() == [[11, 12, 13, 14]]


def test_cross_entropy_loss_is_near_zero_for_a_confident_correct_prediction():
    vocab = 4
    logits = torch.zeros(1, 1, vocab)
    logits[0, 0, 2] = 50.0  # overwhelmingly predicts token 2
    labels = torch.tensor([[2]])
    loss = cross_entropy_loss(logits, labels)
    assert loss.item() < 1e-3


def test_cross_entropy_loss_ignores_masked_positions():
    vocab = 4
    logits = torch.randn(1, 2, vocab)
    labels_all_masked = torch.tensor([[-100, -100]])
    loss = cross_entropy_loss(logits, labels_all_masked)
    assert torch.isnan(loss)  # no positions to average over; matches torch's ignore_index convention
