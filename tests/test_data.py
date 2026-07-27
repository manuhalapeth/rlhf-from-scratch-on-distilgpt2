"""Unit tests for dataset construction, templating, and batching.

None of these touch a model or the network, so they run in well under a
second and are a fast way to check the plumbing before spending minutes
on a full pipeline run against real DistilGPT2.
"""
from rlhf.data import (
    apply_template,
    build_labels,
    build_synthetic_instruction_dataset,
    build_synthetic_preference_dataset,
    collate_lm_batch,
    format_example,
    format_preference,
    iterate_minibatches,
    make_attention_mask,
    mask_prompt_labels,
    pad_batch,
    train_val_split,
)


def test_format_example_contains_prompt_and_response():
    example = {"prompt": "What is 1 + 1?", "response": "It is 2."}
    text = format_example(example)
    assert "### Instruction:" in text
    assert "### Response:" in text
    assert example["prompt"] in text
    assert example["response"] in text


def test_apply_template_maps_every_example():
    examples = build_synthetic_instruction_dataset()
    texts = apply_template(examples)
    assert len(texts) == len(examples)
    assert all(isinstance(t, str) for t in texts)


def test_mask_prompt_labels_ignores_only_prompt_span():
    labels = [10, 11, 12, 13, 14]
    masked = mask_prompt_labels(labels, prompt_length=2)
    assert masked == [-100, -100, 12, 13, 14]


def test_mask_prompt_labels_clamps_to_sequence_length():
    labels = [10, 11]
    masked = mask_prompt_labels(labels, prompt_length=99)
    assert masked == [-100, -100]


def test_build_labels_is_an_independent_copy():
    ids = [1, 2, 3]
    labels = build_labels(ids)
    labels[0] = 999
    assert ids[0] == 1


def test_pad_batch_right_pads_to_longest():
    padded = pad_batch([[1, 2, 3], [4, 5]], pad_id=0)
    assert padded == [[1, 2, 3], [4, 5, 0]]


def test_make_attention_mask_marks_pad_positions():
    mask = make_attention_mask([[1, 2, 0], [4, 0, 0]], pad_id=0)
    assert mask == [[1, 1, 0], [1, 0, 0]]


def test_collate_lm_batch_shapes_match():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ]
    collated = collate_lm_batch(batch, pad_id=0)
    assert collated["input_ids"].shape == (2, 3)
    assert collated["labels"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    assert collated["labels"][1, 2].item() == -100  # padded position stays ignored


def test_iterate_minibatches_covers_every_example_exactly_once():
    examples = list(range(10))
    seen = []
    for mb in iterate_minibatches(examples, batch_size=3, seed=0):
        seen.extend(mb)
    assert sorted(seen) == examples


def test_iterate_minibatches_is_deterministic_given_a_seed():
    examples = list(range(10))
    first = list(iterate_minibatches(examples, batch_size=3, seed=7))
    second = list(iterate_minibatches(examples, batch_size=3, seed=7))
    assert first == second


def test_train_val_split_sizes_and_disjointness():
    examples = build_synthetic_instruction_dataset()
    train, val = train_val_split(examples, val_ratio=0.25, seed=0)
    assert len(train) + len(val) == len(examples)
    train_prompts = {e["prompt"] for e in train}
    val_prompts = {e["prompt"] for e in val}
    assert train_prompts.isdisjoint(val_prompts)


def test_build_synthetic_preference_dataset_repeats_the_pool_deterministically():
    a = build_synthetic_preference_dataset(num_examples=8, seed=0)
    b = build_synthetic_preference_dataset(num_examples=8, seed=0)
    assert a == b
    assert len(a) == 8


def test_format_preference_prefixes_the_prompt():
    example = {"prompt": "Q", "chosen": "good answer", "rejected": "bad answer"}
    formatted = format_preference(example)
    assert formatted["chosen_text"] == "Q good answer"
    assert formatted["rejected_text"] == "Q bad answer"
