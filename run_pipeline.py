"""End to end run of the RLHF from scratch pipeline on real DistilGPT2.

Executes, in order: baseline decoding, supervised finetuning with correct
prompt masking, a LoRA correctness demo trained with a hand rolled
optimizer, reward modeling with a frozen backbone, PPO, DPO with an IPO,
KTO, ORPO, and SimPO loss comparison on the same batch, a win rate
evaluation split into in domain and out of domain prompts, and a minimal
chat and streaming demo.

Every stage records its metrics into `results`, which is written to
assets/results.json and rendered into assets/plots/*.png at the end of
the run. README.md, docs/ARCHITECTURE.md, docs/TRADEOFFS.md, and
docs/RESULTS.md were written against one such run; rerunning this script
will produce numbers in the same neighborhood but not identical ones,
since the synthetic data and small step counts leave real variance run
to run.

Usage: python run_pipeline.py
"""
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlhf import (
    adamw_update,
    batch_sequence_logprob,
    build_eval_prompt_set,
    build_labels,
    build_synthetic_instruction_dataset,
    build_synthetic_preference_dataset,
    chat,
    clip_grad_norm,
    accumulate_gradients,
    collate_lm_batch,
    count_trainable_params,
    dpo_loss,
    evaluate_loss,
    format_example,
    format_preference,
    freeze_base_params,
    gae_advantages,
    generate_and_decode,
    generate_completions,
    init_lora_weights,
    iterate_minibatches,
    ipo_loss,
    kl_penalized_reward,
    kto_loss,
    linear_warmup_schedule,
    load_distilgpt2_model,
    load_distilgpt2_tokenizer,
    lora_linear_forward,
    make_attention_mask,
    mask_prompt_labels,
    merge_lora,
    orpo_loss,
    pad_batch,
    per_token_kl,
    policy_ratio,
    ppo_loss,
    reward_train_step,
    sample_with_temperature,
    score_with_reward,
    set_pad_token_to_eos,
    sft_train_step,
    simpo_loss,
    stream_tokens,
    tokenize_example,
    top_k_filter,
    train_val_split,
    win_rate,
)
import plotting

MODEL_NAME = "distilgpt2"  # real DistilGPT2; sshleifer/tiny-gpt2 has n_embd=2 and cannot learn anything
REPO_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = REPO_ROOT / "assets" / "plots"
RESULTS_PATH = REPO_ROOT / "assets" / "results.json"


def _token_logps(logits, token_ids):
    """Per token log probability of each selected token, no summation. Local runner glue."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def _response_mask_and_ids(tokenizer, prompt_prefix, full_text, max_length):
    """Tokenize full_text and derive (ids, labels, response_mask) with the prompt
    portion (prompt_prefix) masked out, reusing mask_prompt_labels."""
    prompt_len = len(tokenizer.encode(prompt_prefix))
    ids = tokenize_example(tokenizer, full_text, max_length=max_length)
    labels = mask_prompt_labels(build_labels(ids), prompt_len)
    resp_mask = [0 if l == -100 else 1 for l in labels]
    return ids, labels, resp_mask


def freeze_and_eval(m):
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


class _HiddenBackbone:
    """Adapter: reward_train_step expects a callable returning a hidden state
    tensor of shape (B, T, H), but we have a full causal LM model."""
    def __init__(self, m):
        self.m = m

    def __call__(self, ids, attention_mask=None):
        out = self.m(ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-1]


def build_in_domain_eval_prompt_set():
    """Prompts that mirror the six preference dataset topics (geography, arithmetic,
    a science explanation, a coding how to, a fairy tale summary, a science fact)
    without repeating the literal training strings. Used to separate 'did the
    model learn the six preference pairs by heart' from 'did it generalize to
    unrelated topics', see docs/RESULTS.md."""
    return [
        "What is the capital of Italy?",
        "What is 3 + 5?",
        "Explain what gravity is.",
        "How do I sort a list in Python?",
        "Summarize the plot of Snow White in one sentence.",
        "What is the freezing point of water at sea level in Celsius?",
    ]


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {"model_name": MODEL_NAME}

    # ------------------------------------------------------------------
    # 1) Tokenizer + base model
    # ------------------------------------------------------------------
    tokenizer = load_distilgpt2_tokenizer(MODEL_NAME)
    set_pad_token_to_eos(tokenizer)
    model = load_distilgpt2_model(MODEL_NAME)
    pad_id = tokenizer.pad_token_id
    hidden = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
    print(f"Loaded {MODEL_NAME}; vocab={len(tokenizer)}, hidden={hidden}, pad==eos? {tokenizer.pad_token == tokenizer.eos_token}")

    # ------------------------------------------------------------------
    # 2) Baseline generation (unaligned, untrained)
    # ------------------------------------------------------------------
    base_out = generate_and_decode(model, tokenizer, "Hello, how are you?", max_new_tokens=12)
    print("Base completion:", repr(base_out))
    results["base_completion"] = base_out

    # ------------------------------------------------------------------
    # 3) SFT: build and tokenize data with correct prompt token masking
    # ------------------------------------------------------------------
    sft_data = build_synthetic_instruction_dataset()
    train_data, val_data = train_val_split(sft_data, val_ratio=0.25, seed=0)

    def make_sft_batches(examples, bs=2, seed=0):
        built = []
        for ex in examples:
            prompt_prefix = format_example({"prompt": ex["prompt"], "response": ""})
            full_text = format_example(ex)
            ids, labels, _ = _response_mask_and_ids(tokenizer, prompt_prefix, full_text, max_length=64)
            built.append({"input_ids": ids, "labels": labels})
        for mb in iterate_minibatches(built, batch_size=bs, seed=seed):
            yield collate_lm_batch(mb, pad_id)

    print("\n=== SFT ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    sft_epochs = 12
    sft_losses = []
    step = 0
    for epoch in range(sft_epochs):
        for batch in make_sft_batches(train_data, bs=2, seed=epoch):
            warmup_mult = linear_warmup_schedule(step, warmup_steps=5)
            for group in optimizer.param_groups:
                group["lr"] = 5e-5 * warmup_mult
            loss = sft_train_step(model, batch, optimizer)
            sft_losses.append(float(loss))
            step += 1
    sft_val_loss = float(evaluate_loss(model, list(make_sft_batches(val_data, bs=2))))
    print(f"SFT loss: {sft_losses[0]:.3f} -> {sft_losses[-1]:.3f} over {len(sft_losses)} steps; val_loss={sft_val_loss:.3f}")
    sft_out = generate_and_decode(model, tokenizer, "What is the capital of Japan?", max_new_tokens=12)
    print("Post SFT completion:", repr(sft_out))
    results["sft"] = {"train_losses": sft_losses, "val_loss": sft_val_loss, "post_sft_completion": sft_out}

    # ------------------------------------------------------------------
    # 4) LoRA correctness demo, trained with the from scratch optimizer
    #    primitives instead of torch.optim, against the real lm_head
    #    weight of the finetuned model. Isolated demo: does not feed
    #    forward into later stages.
    # ------------------------------------------------------------------
    print("\n=== LoRA (from scratch optimizer) ===")
    lm_head_weight = model.lm_head.weight.detach().clone()  # (vocab, n_embd), frozen base weight
    r, alpha = 8, 16
    A, B = init_lora_weights(in_features=hidden, out_features=len(tokenizer), r=r, seed=0)
    A.requires_grad_(True)
    B.requires_grad_(True)

    freeze_base_params(model)  # every model.* param name lacks "lora" -> all frozen
    base_trainable = count_trainable_params(model)
    lora_trainable = A.numel() + B.numel()
    trainable_pct = 100 * lora_trainable / (lora_trainable + sum(p.numel() for p in model.parameters()))
    print(f"Trainable params: base model={base_trainable}, LoRA adapter={lora_trainable} ({trainable_pct:.3f}% of total)")

    probes = []
    for ex in (train_data[0], train_data[1 % len(train_data)]):
        probe_ids = torch.tensor([tokenize_example(tokenizer, format_example(ex), max_length=32)])
        with torch.no_grad():
            hidden_states = model(probe_ids, output_hidden_states=True).hidden_states[-1][0]
        probes.append((hidden_states[:-1], probe_ids[0, 1:]))

    with torch.no_grad():
        x0, _ = probes[0]
        logits_zero_delta = lora_linear_forward(x0, lm_head_weight, A, B, alpha, r)
        base_logits = F.linear(x0, lm_head_weight)
        b_zero_matches_base = bool(torch.allclose(logits_zero_delta, base_logits))
        print(f"B initialized to zero -> LoRA logits == base logits: {b_zero_matches_base}")

    opt_state = {"A": {}, "B": {}}
    lora_losses = []
    total_lora_steps = 15
    for lora_step in range(total_lora_steps):
        micro_grads_A, micro_grads_B, micro_losses = [], [], []
        for x, labels in probes:
            logits = lora_linear_forward(x, lm_head_weight, A, B, alpha, r)
            loss = F.cross_entropy(logits, labels)
            micro_losses.append(float(loss))
            grad_A, grad_B = torch.autograd.grad(loss, [A, B])
            micro_grads_A.append(grad_A)
            micro_grads_B.append(grad_B)
        lora_losses.append(sum(micro_losses) / len(micro_losses))

        grad_A = accumulate_gradients(micro_grads_A)
        grad_B = accumulate_gradients(micro_grads_B)
        clip_grad_norm([grad_A, grad_B], max_norm=1.0)
        warmup_mult = linear_warmup_schedule(lora_step, warmup_steps=5)
        lr = 0.01 * warmup_mult
        with torch.no_grad():
            adamw_update(A, grad_A, opt_state["A"], lr=lr)
            adamw_update(B, grad_B, opt_state["B"], lr=lr)

    print(f"LoRA adapter loss (from scratch AdamW plus grad accumulation): {lora_losses[0]:.3f} -> {lora_losses[-1]:.3f}")

    merged_weight = merge_lora(lm_head_weight, A.detach(), B.detach(), scaling=alpha / r)
    with torch.no_grad():
        x0, _ = probes[0]
        merged_logits = F.linear(x0, merged_weight)
        lora_logits = lora_linear_forward(x0, lm_head_weight, A, B, alpha, r)
        merge_matches = bool(torch.allclose(merged_logits, lora_logits, atol=1e-5))
        print(f"merge_lora matches lora_linear_forward: {merge_matches}")

    for p in model.parameters():
        p.requires_grad_(True)  # undo freeze_base_params before continuing to train `model` downstream

    results["lora"] = {
        "base_trainable": int(base_trainable),
        "lora_trainable": int(lora_trainable),
        "trainable_pct": trainable_pct,
        "b_zero_matches_base": b_zero_matches_base,
        "losses": lora_losses,
        "merge_matches_forward": merge_matches,
    }

    # ------------------------------------------------------------------
    # 5) Reward model: frozen backbone snapshot plus a trained scalar head
    # ------------------------------------------------------------------
    print("\n=== Reward model ===")
    rm_backbone_model = freeze_and_eval(copy.deepcopy(model))
    backbone = _HiddenBackbone(rm_backbone_model)

    reward_head = torch.nn.Linear(hidden, 1)
    rm_opt = torch.optim.AdamW(reward_head.parameters(), lr=1e-3)

    def build_rm_batch(pref, max_length=48):
        chosen_texts, rejected_texts = [], []
        for ex in pref:
            formatted = format_preference(ex)
            chosen_texts.append(formatted["chosen_text"])
            rejected_texts.append(formatted["rejected_text"])
        ce = tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        re_ = tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return {
            "chosen_input_ids": ce["input_ids"], "chosen_attention_mask": ce["attention_mask"],
            "rejected_input_ids": re_["input_ids"], "rejected_attention_mask": re_["attention_mask"],
        }

    pref_data = build_synthetic_preference_dataset(num_examples=12, seed=0)
    rm_losses, rm_accs = [], []
    for epoch in range(5):
        for mb in iterate_minibatches(pref_data, batch_size=4, seed=epoch):
            rm_batch = build_rm_batch(mb)
            rm_out = reward_train_step(backbone, reward_head, rm_batch, rm_opt)
            rm_losses.append(rm_out["loss"])
            rm_accs.append(rm_out["accuracy"])
    print(f"RM loss: {rm_losses[0]:.3f} -> {rm_losses[-1]:.3f}; pairwise accuracy: {rm_accs[0]:.2f} -> {rm_accs[-1]:.2f}")
    results["reward_model"] = {"losses": rm_losses, "accuracies": rm_accs}

    reward_bundle = {"model": rm_backbone_model, "weight": reward_head.weight, "bias": reward_head.bias}

    # Raw RM logits are not on any particular scale; standardize against the
    # training distribution so downstream KL penalized rewards (PPO) are well
    # conditioned instead of swamping the value loss with huge targets.
    _ref_scores = []
    for ex in pref_data:
        formatted = format_preference(ex)
        _ref_scores.append(score_with_reward(reward_bundle, tokenizer, "", formatted["chosen_text"]))
        _ref_scores.append(score_with_reward(reward_bundle, tokenizer, "", formatted["rejected_text"]))
    reward_mean = float(np.mean(_ref_scores))
    reward_std = float(np.std(_ref_scores)) + 1e-6
    print(f"Reward normalization stats: mean={reward_mean:.3f}, std={reward_std:.3f}")
    results["reward_model"]["normalization_mean"] = reward_mean
    results["reward_model"]["normalization_std"] = reward_std

    def normalized_reward(prompt, completion):
        raw = score_with_reward(reward_bundle, tokenizer, prompt, completion)
        return (raw - reward_mean) / reward_std

    # ------------------------------------------------------------------
    # 6) PPO: sample rollouts from the policy, score with the frozen RM,
    #    KL penalize against a frozen reference, and update with the
    #    clipped surrogate plus value plus entropy objective.
    # ------------------------------------------------------------------
    print("\n=== PPO ===")
    ppo_policy = copy.deepcopy(model)
    ppo_ref = freeze_and_eval(copy.deepcopy(model))
    value_head = torch.nn.Linear(hidden, 1)
    ppo_opt = torch.optim.AdamW(list(ppo_policy.parameters()) + list(value_head.parameters()), lr=1e-5)

    def rollout(prompt_text, max_new_tokens=12, temperature=0.8, top_k=50):
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        generated = prompt_ids.clone()
        ppo_policy.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                next_logits = ppo_policy(generated).logits[0, -1, :]
                next_logits = top_k_filter(next_logits, top_k)
                next_id = sample_with_temperature(next_logits, temperature)
                if next_id == tokenizer.eos_token_id:
                    break
                generated = torch.cat([generated, torch.tensor([[next_id]])], dim=1)
        ppo_policy.train()
        return prompt_ids, generated[:, prompt_ids.shape[1]:]

    ppo_prompts = [ex["prompt"] for ex in build_synthetic_instruction_dataset()[:3]]
    beta_kl = 0.05
    ppo_policy_losses, ppo_value_losses, ppo_entropies, ppo_rewards = [], [], [], []

    for prompt_text in ppo_prompts:
        prompt_ids, response_ids = rollout(prompt_text)
        if response_ids.shape[1] < 2:
            continue
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attn = torch.ones_like(full_ids)
        resp_slice = slice(prompt_ids.shape[1] - 1, full_ids.shape[1] - 1)

        with torch.no_grad():
            ref_logits = ppo_ref(full_ids, attention_mask=attn).logits[:, :-1, :]
            ref_token_logps = _token_logps(ref_logits, full_ids[:, 1:])[0, resp_slice]

            old_logits = ppo_policy(full_ids, attention_mask=attn).logits[:, :-1, :]
            old_token_logps = _token_logps(old_logits, full_ids[:, 1:])[0, resp_slice]

        kl = per_token_kl(old_token_logps, ref_token_logps)
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        reward_scalar = normalized_reward(prompt_text, response_text)
        ppo_rewards.append(reward_scalar)

        rewards = [kl_penalized_reward(0.0, float(kl[t]), beta_kl) for t in range(len(kl) - 1)]
        rewards.append(kl_penalized_reward(reward_scalar, float(kl[-1]), beta_kl))
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        for _inner_epoch in range(2):
            out = ppo_policy(full_ids, attention_mask=attn, output_hidden_states=True)
            logits = out.logits[:, :-1, :]
            hidden_states = out.hidden_states[-1][:, :-1, :]
            new_resp_logps = _token_logps(logits, full_ids[:, 1:])[0, resp_slice]
            values = torch.cat([value_head(hidden_states).squeeze(-1)[0, resp_slice], torch.zeros(1)])

            with torch.no_grad():
                advantages = gae_advantages(rewards_t, values.detach(), gamma=0.99, lam=0.95)
                returns = advantages + values.detach()[:-1]

            ratio = policy_ratio(new_resp_logps, old_token_logps.detach())
            out_dict = ppo_loss(ratio, advantages, values[:-1], returns, logits[0, resp_slice],
                                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
            ppo_opt.zero_grad()
            out_dict["loss"].backward()
            ppo_opt.step()
            ppo_policy_losses.append(float(out_dict["policy_loss"]))
            ppo_value_losses.append(float(out_dict["value_loss"]))
            ppo_entropies.append(float(out_dict["entropy"]))

    print(f"PPO normalized rewards per rollout: {[round(r, 3) for r in ppo_rewards]}")
    print(f"PPO policy_loss: {ppo_policy_losses[0]:.3f} -> {ppo_policy_losses[-1]:.3f}")
    print(f"PPO value_loss:  {ppo_value_losses[0]:.3f} -> {ppo_value_losses[-1]:.3f}")
    print(f"PPO entropy:     {ppo_entropies[0]:.3f} -> {ppo_entropies[-1]:.3f}")
    results["ppo"] = {
        "rewards": ppo_rewards,
        "policy_losses": ppo_policy_losses,
        "value_losses": ppo_value_losses,
        "entropies": ppo_entropies,
    }

    del ppo_ref, ppo_policy, value_head
    gc.collect()

    # ------------------------------------------------------------------
    # 7) DPO: preference optimize a fresh copy of the SFT model against a
    #    frozen reference, then compare IPO, KTO, ORPO, and SimPO losses
    #    on the same final batch.
    # ------------------------------------------------------------------
    print("\n=== DPO ===")
    dpo_policy = copy.deepcopy(model)
    dpo_ref = freeze_and_eval(copy.deepcopy(model))
    dpo_opt = torch.optim.AdamW(dpo_policy.parameters(), lr=5e-6)

    def build_dpo_group(examples, key, max_length=48):
        ids_list, mask_list, lengths = [], [], []
        for ex in examples:
            prompt_prefix = ex["prompt"] + " "
            full_text = prompt_prefix + ex[key]
            ids, _, resp_mask = _response_mask_and_ids(tokenizer, prompt_prefix, full_text, max_length)
            ids_list.append(ids)
            mask_list.append(resp_mask)
            lengths.append(sum(resp_mask))
        padded_ids = pad_batch(ids_list, pad_id)
        padded_mask = pad_batch(mask_list, 0)
        attn = make_attention_mask(padded_ids, pad_id)
        return (torch.tensor(padded_ids), torch.tensor(attn), torch.tensor(padded_mask),
                torch.tensor(lengths, dtype=torch.float32))

    def seq_logps(m, ids, attn, resp_mask, grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = m(ids, attention_mask=attn).logits[:, :-1, :]
            labels = ids[:, 1:]
            mask = (resp_mask[:, 1:] * attn[:, 1:]).to(logits.dtype)
            return batch_sequence_logprob(logits, labels, attention_mask=mask)

    pref_data_dpo = build_synthetic_preference_dataset(num_examples=12, seed=1)
    dpo_losses, dpo_margins = [], []
    last_batch_logps = None
    for epoch in range(4):
        for mb in iterate_minibatches(pref_data_dpo, batch_size=4, seed=epoch):
            c_ids, c_attn, c_mask, c_len = build_dpo_group(mb, "chosen")
            r_ids, r_attn, r_mask, r_len = build_dpo_group(mb, "rejected")

            pc = seq_logps(dpo_policy, c_ids, c_attn, c_mask, grad=True)
            pr = seq_logps(dpo_policy, r_ids, r_attn, r_mask, grad=True)
            with torch.no_grad():
                rc = seq_logps(dpo_ref, c_ids, c_attn, c_mask, grad=False)
                rr = seq_logps(dpo_ref, r_ids, r_attn, r_mask, grad=False)

            loss = dpo_loss(pc, pr, rc, rr, beta=0.1)
            dpo_opt.zero_grad()
            loss.backward()
            dpo_opt.step()

            dpo_losses.append(float(loss))
            dpo_margins.append(float(((pc - pr) - (rc - rr)).mean()))
            last_batch_logps = (pc.detach(), pr.detach(), rc.detach(), rr.detach(), c_len, r_len)

    print(f"DPO loss: {dpo_losses[0]:.3f} -> {dpo_losses[-1]:.3f}")
    print(f"DPO preference margin (policy vs ref): {dpo_margins[0]:.3f} -> {dpo_margins[-1]:.3f}")

    pc, pr, rc, rr, c_len, r_len = last_batch_logps
    sft_loss_chosen = -pc.mean() / c_len.mean()
    preference_loss_comparison = {
        "dpo": float(dpo_loss(pc, pr, rc, rr, beta=0.1)),
        "ipo": float(ipo_loss(pc, pr, rc, rr, beta=0.1)),
    }
    kto_logps = torch.cat([pc, pr])
    kto_ref = torch.cat([rc, rr])
    kto_labels = torch.cat([torch.ones_like(pc), torch.zeros_like(pr)])
    preference_loss_comparison["kto"] = float(kto_loss(kto_logps, kto_ref, kto_labels, beta=0.1))
    preference_loss_comparison["orpo"] = float(orpo_loss(pc / c_len, pr / r_len, sft_loss_chosen, lambda_or=0.1))
    preference_loss_comparison["simpo"] = float(simpo_loss(pc, pr, c_len, r_len, beta=2.0, gamma=0.5))

    print("\n--- Preference loss comparison on the final DPO batch (DPO is the one actually trained above) ---")
    for name, value in preference_loss_comparison.items():
        print(f"{name}_loss: {value:.4f}")

    results["dpo"] = {
        "losses": dpo_losses,
        "margins": dpo_margins,
        "preference_loss_comparison": preference_loss_comparison,
    }

    del dpo_ref
    gc.collect()

    # ------------------------------------------------------------------
    # 8) Eval tooling: win rate of the DPO aligned model vs. a pristine,
    #    never trained base model, judged by the frozen reward model,
    #    split into in domain and out of domain prompt sets.
    # ------------------------------------------------------------------
    print("\n=== Eval: aligned vs. base ===")
    unaligned_baseline = load_distilgpt2_model(MODEL_NAME)
    eval_sets = {
        "out_of_domain": build_eval_prompt_set(),
        "in_domain": build_in_domain_eval_prompt_set(),
    }
    eval_results = {}
    for split_name, eval_prompts in eval_sets.items():
        comps_aligned = generate_completions(dpo_policy, tokenizer, eval_prompts, max_new_tokens=20)
        comps_base = generate_completions(unaligned_baseline, tokenizer, eval_prompts, max_new_tokens=20)

        print(f"\n[{split_name}]")
        examples = []
        for p, a, b in zip(eval_prompts, comps_aligned, comps_base):
            print(f"  prompt: {p!r}\n    aligned: {a[len(p):]!r}\n    base:    {b[len(p):]!r}")
            examples.append({"prompt": p, "aligned": a[len(p):], "base": b[len(p):]})

        rate = win_rate(reward_bundle, tokenizer, eval_prompts, comps_aligned, comps_base)
        print(f"{split_name} win rate (reward model judged): {rate:.2f} over {len(eval_prompts)} prompts")
        eval_results[split_name] = {"win_rate": rate, "examples": examples}

    results["eval"] = eval_results

    # ------------------------------------------------------------------
    # 9) Minimal chat interface (batch reply plus token streaming)
    # ------------------------------------------------------------------
    print("\n=== Chat ===")
    reply = chat(dpo_policy, tokenizer, "Say hi.", system_prompt="You are helpful.", max_new_tokens=20)
    print("Chat reply:", repr(reply))

    print("Streaming reply: ", end="", flush=True)
    streamed_pieces = []
    for piece in stream_tokens(dpo_policy, tokenizer, "The capital of France is", max_new_tokens=12):
        print(piece, end="", flush=True)
        streamed_pieces.append(piece)
    print()
    results["chat"] = {"reply": reply, "streaming_reply": "".join(streamed_pieces)}

    # ------------------------------------------------------------------
    # 10) Persist metrics and render plots
    # ------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")

    plotting.plot_sft_loss(sft_losses, sft_val_loss, PLOTS_DIR / "sft_loss.png")
    plotting.plot_lora_loss(lora_losses, PLOTS_DIR / "lora_loss.png")
    plotting.plot_reward_model(rm_losses, rm_accs, PLOTS_DIR / "reward_model.png")
    plotting.plot_ppo_training(ppo_policy_losses, ppo_value_losses, ppo_entropies, PLOTS_DIR / "ppo_training.png")
    plotting.plot_dpo_training(dpo_losses, dpo_margins, PLOTS_DIR / "dpo_training.png")
    plotting.plot_preference_loss_comparison(preference_loss_comparison, PLOTS_DIR / "preference_loss_comparison.png")
    plotting.plot_win_rate_comparison(
        {"in domain": eval_results["in_domain"]["win_rate"], "out of domain": eval_results["out_of_domain"]["win_rate"]},
        PLOTS_DIR / "win_rate_comparison.png",
    )
    print(f"Wrote plots to {PLOTS_DIR}")
