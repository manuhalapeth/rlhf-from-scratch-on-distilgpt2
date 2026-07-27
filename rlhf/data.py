"""Synthetic datasets, prompt templates, and batching utilities.

Everything here is pure Python and tensor plumbing with no model calls,
which is what makes it fast enough to unit test directly.
"""
import math
import random

import torch


def build_synthetic_instruction_dataset():
    """Return a small, deterministic in memory instruction and response dataset."""
    return [
        {
            "prompt": "Translate the following sentence to French: 'Good morning.'",
            "response": "Bonjour.",
        },
        {
            "prompt": "Summarize the plot of Romeo and Juliet in one sentence.",
            "response": "Two young lovers from feuding families in Verona die tragically, "
                         "ultimately reconciling their households.",
        },
        {
            "prompt": "What is the capital of Japan?",
            "response": "The capital of Japan is Tokyo.",
        },
        {
            "prompt": "Write a short poem about the ocean.",
            "response": "The ocean waves in endless blue, "
                         "carry secrets old and new.",
        },
        {
            "prompt": "Explain what a for loop does in Python.",
            "response": "A for loop repeats a block of code once for each item "
                         "in a sequence, such as a list or range of numbers.",
        },
        {
            "prompt": "Give a synonym for 'happy'.",
            "response": "Joyful.",
        },
    ]


def format_example(example):
    """Render a prompt/response dict into one training string."""
    return "### Instruction:\n{}\n\n### Response:\n{}".format(
        example["prompt"], example["response"]
    )


def apply_template(examples):
    """Map format_example over a list of instruction examples."""
    return [format_example(example) for example in examples]


def tokenize_example(tokenizer, text, max_length=64):
    """Encode text into token ids, truncated to max_length, no padding."""
    return tokenizer.encode(text, truncation=True, max_length=max_length)


def build_labels(input_ids):
    """Return an independent copy of input_ids to serve as next token labels."""
    return list(input_ids)


def mask_prompt_labels(labels, prompt_length):
    """Replace the first prompt_length label positions with the ignore index (negative one hundred)."""
    n = min(prompt_length, len(labels))
    return [-100] * n + list(labels[n:])


def pad_batch(sequences, pad_id):
    """Right pad a list of token id sequences to the longest length."""
    if not sequences:
        return []
    max_len = max(len(seq) for seq in sequences)
    return [list(seq) + [pad_id] * (max_len - len(seq)) for seq in sequences]


def make_attention_mask(padded_ids, pad_id):
    """Return a same shape zero/one mask with one where token is not pad_id."""
    return [[1 if token != pad_id else 0 for token in seq] for seq in padded_ids]


def collate_lm_batch(batch, pad_id):
    """Turn a list of tokenized examples into a single batched dict for the model."""
    input_ids_list = [example["input_ids"] for example in batch]
    labels_list = [example["labels"] for example in batch]

    padded_input_ids = pad_batch(input_ids_list, pad_id)
    padded_labels = pad_batch(labels_list, -100)
    attention_mask = make_attention_mask(padded_input_ids, pad_id)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def iterate_minibatches(examples, batch_size, seed=0):
    """Yield successive minibatches from a deterministically shuffled copy of examples."""
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]


def train_val_split(examples, val_ratio=0.2, seed=0):
    """Deterministically split examples into train and validation sets using a seeded shuffle."""
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_size = math.floor(len(shuffled) * val_ratio)
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    return train, val


def build_synthetic_preference_dataset(num_examples=4, seed=0):
    """Return a reproducible list of prompt/chosen/rejected dicts."""
    pool = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "I do not know.",
        },
        {
            "prompt": "What is 2 + 2?",
            "chosen": "2 + 2 equals 4.",
            "rejected": "2 + 2 equals 5.",
        },
        {
            "prompt": "Explain what photosynthesis is.",
            "chosen": "Photosynthesis is the process by which plants convert sunlight, "
                       "water, and carbon dioxide into glucose and oxygen.",
            "rejected": "It's a thing plants do I think, not really sure honestly.",
        },
        {
            "prompt": "How do I reverse a list in Python?",
            "chosen": "You can reverse a list in Python using `my_list[::-1]` or the "
                       "in place `my_list.reverse()` method.",
            "rejected": "Just Google it, I'm not sure how that works.",
        },
        {
            "prompt": "Summarize the plot of Cinderella in one sentence.",
            "chosen": "A kind young woman overcomes her cruel stepfamily with the help "
                       "of magic to find happiness with a prince.",
            "rejected": "Something about a girl and shoes I guess.",
        },
        {
            "prompt": "What is the boiling point of water at sea level in Celsius?",
            "chosen": "Water boils at 100 degrees Celsius at sea level.",
            "rejected": "Water boils at some temperature, maybe 50 degrees or so.",
        },
    ]

    if num_examples <= 0:
        return []

    n = len(pool)
    shift = seed % n
    rotated = pool[shift:] + pool[:shift]

    selected = []
    while len(selected) < num_examples:
        selected.extend(rotated)
    selected = selected[:num_examples]

    return [dict(example) for example in selected]


def format_preference(example):
    """Render a preference example into chosen_text and rejected_text strings."""
    return {
        "chosen_text": example["prompt"] + " " + example["chosen"],
        "rejected_text": example["prompt"] + " " + example["rejected"],
    }
