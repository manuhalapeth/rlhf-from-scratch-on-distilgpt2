"""Tokenizer/model loading, decoding strategies, and the chat surface.

Covers greedy decoding, temperature sampling, top k and top p (nucleus)
filtering, token streaming, and the minimal chat wrapper used to compare
an aligned model against its unaligned starting point.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    """Load a Hugging Face tokenizer for a DistilGPT2 compatible model.

    Parameters
    ----------
    model_name : str
        Name (or path) of the model/tokenizer on the Hugging Face Hub.
        Defaults to a tiny stand in model so it runs quickly on CPU; pass
        "distilgpt2" for the real model with actual language modeling
        capacity.

    Returns
    -------
    PreTrainedTokenizerBase
        The loaded tokenizer instance, ready to encode/decode text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    """Load a causal language model from the Hugging Face Hub, ready for inference.

    Parameters
    ----------
    model_name : str
        Name (or path) of the model on the Hugging Face Hub.

    Returns
    -------
    PreTrainedModel
        The loaded causal LM in evaluation mode (dropout disabled),
        suitable for forward passes and `.generate()` calls.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model


def set_pad_token_to_eos(tokenizer):
    """Assign the tokenizer's EOS token as its pad token, in place."""
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    """Tokenize a prompt, greedily generate a continuation, and decode it to text."""
    if tokenizer.pad_token is None:
        set_pad_token_to_eos(tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def greedy_decode(logits):
    """Return the argmax token id from a single row logits vector."""
    if logits.dim() == 2:
        if logits.size(0) != 1:
            raise ValueError(f"Expected a single row 2D tensor of shape (1, vocab), got shape {tuple(logits.shape)}")
        logits = logits[0]
    elif logits.dim() != 1:
        raise ValueError(f"Expected a 1D or 2D tensor, got shape {tuple(logits.shape)}")
    return int(torch.argmax(logits).item())


def sample_with_temperature(logits, temperature):
    """Draw a single token id from logits after temperature scaling."""
    logits = logits.reshape(-1).float()
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return int(token_id.item())


def top_k_filter(logits, k):
    """Keep the k largest entries of logits, set the rest to negative infinity."""
    logits = logits.clone()
    vocab_size = logits.size(-1)
    if k <= 0:
        return torch.full_like(logits, float("-inf"))
    k = min(k, vocab_size)
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[..., -1]
    mask = logits < threshold
    logits[mask] = float("-inf")
    return logits


def top_p_filter(logits, p):
    """Mask logits outside the smallest cumulative probability nucleus."""
    logits = torch.as_tensor(logits, dtype=torch.float).clone().reshape(-1)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_mask[0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[sorted_indices] = sorted_mask
    logits[mask] = float("-inf")
    return logits


def build_eval_prompt_set():
    """Return a held out set of short, diverse instruction style prompts for win rate evaluation."""
    return [
        "Explain the water cycle in simple terms.",
        "Write a short thank you note to a coworker.",
        "What are three tips for staying productive while working from home?",
        "Summarize the benefits of regular exercise.",
        "Give step by step instructions for making a cup of tea.",
        "Recommend a book for someone who enjoys mystery novels.",
    ]


def generate_completions(model, tokenizer, prompts, max_new_tokens=8):
    """Generate a decoded completion for each prompt, reusing generate_and_decode."""
    return [
        generate_and_decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        for prompt in prompts
    ]


def stream_tokens(model, tokenizer, prompt, max_new_tokens=8):
    """Yield newly generated text pieces one token at a time via greedy decoding."""
    if tokenizer.pad_token is None:
        set_pad_token_to_eos(tokenizer)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prev_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = greedy_decode(next_token_logits)

            next_token_tensor = torch.tensor([[next_token_id]])
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

            new_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            piece = new_text[len(prev_text):]
            prev_text = new_text
            yield piece


def apply_stop_tokens(text, stop_tokens, eos_token=None):
    """Truncate text at the earliest occurrence of any stop token or eos_token."""
    markers = list(stop_tokens)
    if eos_token is not None:
        markers.append(eos_token)

    earliest_idx = None
    for marker in markers:
        if not marker:
            continue
        idx = text.find(marker)
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx

    if earliest_idx is None:
        return text
    return text[:earliest_idx]


def chat(model, tokenizer, user_message, system_prompt=None, max_new_tokens=32, stop_tokens=None):
    """Top level conversational wrapper: build a chat prompt, generate, and return the cleaned reply."""
    tokenizer = set_pad_token_to_eos(tokenizer)

    if system_prompt:
        prompt = f"### System:\n{system_prompt}\n\n### User:\n{user_message}\n\n### Assistant:\n"
    else:
        prompt = f"### User:\n{user_message}\n\n### Assistant:\n"

    if max_new_tokens <= 0:
        return ""

    full_text = generate_and_decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    reply = full_text[len(prompt):]

    if stop_tokens is None:
        stop_tokens = []

    reply = apply_stop_tokens(reply, stop_tokens, eos_token=tokenizer.eos_token)
    return reply.strip()
