# Design tradeoffs

This document is the reasoning behind the choices in the code, written so a
reader can judge whether each one was sensible without having to reverse
engineer it from the diff. Each entry states the decision, the alternative
that was considered, and why the decision won for this project's goal:
building an honest, from scratch, fully wired RLHF pipeline on real, small
scale hardware.

## Real DistilGPT2 instead of a tiny test fixture

The pipeline this project grew out of defaulted to `sshleifer/tiny-gpt2`, a
two layer, two hidden dimension model that exists purely as a fast fixture
for the Hugging Face test suite. With a two dimensional hidden state trying
to represent a 50257 token vocabulary, that model is mathematically
incapable of moving far from the uniform guessing loss of `ln(50257) ≈
10.82`, regardless of how correct the surrounding training code is. Every
number in `docs/RESULTS.md` comes from real DistilGPT2 (six layers, 768
hidden dimensions, about 82 million parameters), which is small enough to
finetune on a laptop CPU in minutes but large enough to actually learn.

## Reward model backbone is a frozen, separate copy

The reward model's backbone could either share the same weights as the
policy being trained (cheaper, and what the pipeline did before this
rewrite) or be an independently frozen snapshot. Sharing is cheaper in
memory but means the reward model's judgment silently drifts as the policy
underneath it keeps training, since the "reward model" is then scoring
completions using hidden states from a moving target. A reward model whose
opinion changes as a side effect of the thing it is judging is not
functioning as an external, fixed judge anymore, which is the entire point
of having one. This project pays the extra memory cost of `copy.deepcopy`
and keeps the reward model backbone frozen from the moment it is snapshotted
right after supervised finetuning.

## PPO and DPO both start from the same post SFT checkpoint, independently

An alternative design would chain the methods: run PPO, then run DPO on top
of the PPO output, and report one final "aligned" model. That produces a
single number but destroys the ability to say which method is responsible
for which change in behavior, and it means a bug in an earlier stage
silently contaminates every later one. This project runs PPO and DPO as two
independent branches from the same frozen starting point (see the diagram in
`docs/ARCHITECTURE.md`), so each method's effect can be read in isolation.
The tradeoff is that neither branch benefits from the other's updates, so
neither is likely to reach an especially strong final policy on its own with
this little data. That is an acceptable cost for a project whose goal is
correctness and legibility, not a leaderboard number.

## Reward normalization before PPO

A reward model trained with a Bradley Terry pairwise loss only has to get
the *ordering* of chosen versus rejected right; nothing constrains the
*scale* of its output logits. In an early run of this pipeline, raw reward
scores on PPO rollouts landed anywhere from negative twelve to negative
eighteen. Fed directly into `kl_penalized_reward` and then into the value
loss target, that scale dominated PPO's total loss and made the printed
numbers look like the optimizer was diverging, when the real issue was
unscaled inputs. The fix, standardizing rewards against the mean and
standard deviation observed over the reward model's own training batch
before using them in PPO, is standard practice in production RLHF
implementations (OpenAI's and TRL's PPO trainers both do a version of this),
not a cosmetic patch specific to this repository. See
`docs/RESULTS.md` for the before and after numbers.

## The LoRA demonstration is isolated, not chained into later stages

`rlhf/lora.py`'s functions are exercised in `run_pipeline.py` against the
real `lm_head` weight of the finetuned model, trained with the from scratch
optimizer in `rlhf/optim.py`, and then verified against `merge_lora`. That
adapter is not merged back into the model that PPO and DPO later train,
which is a deliberate scope decision: correctly wiring a trainable low rank
adapter into every attention and feedforward projection inside a Hugging
Face `GPT2LMHeadModel`, so the whole model can be finetuned adapter only
through PPO and DPO as well, is a real engineering project on its own and
was judged to be out of scope for what this repository is trying to
demonstrate, which is that the LoRA math itself, freezing, delta
computation, gradient based training, and merging, is implemented correctly
and verifiably. The honest label for this section is "a correctness proof
against a real weight matrix," not "a fully productionized adapter
finetuning path," and the code and this document both say so.

## Frozen backbone, trained head only, for the reward model

The reward model in this project finetunes only a single linear head on top
of a frozen DistilGPT2 backbone, rather than finetuning the whole backbone
end to end the way a production reward model typically would. On a dataset
of six repeating preference pairs, finetuning an 82 million parameter
backbone would memorize those six pairs almost immediately and tell you
nothing about generalization; freezing the backbone and training only the
roughly 770 parameter head is both faster and a fairer test of whether the
signal in the frozen representations is separable at all. It reached 100%
pairwise accuracy on this dataset, which says the representations are
separable, not that the reward model would generalize to prompts outside
this synthetic set. See the in domain versus out of domain evaluation split
in `docs/RESULTS.md` for direct evidence of that generalization gap.

## Sparse terminal reward with per token KL shaping for PPO, not a per token reward model

PPO in this repository scores an entire generated completion once, at the
end, with the reward model, and distributes only the KL penalty at every
generated token (the standard reward shaping used in essentially every
public RLHF PPO implementation, OpenAI's included). An alternative would be
to have the reward model score every partial prefix of the completion and
use that as a dense per token reward. Dense reward shaping is more sample
efficient in principle, but it requires a reward model that has actually
been trained to score partial, incomplete text sensibly, which a reward
model trained only on complete chosen and rejected responses (as this one
is) has no guarantee of doing. Sparse terminal reward keeps the reward
model's job identical between training and evaluation: score a finished
response, nothing else.

## Small, synthetic datasets instead of a public preference dataset

Both the instruction dataset (six examples) and the preference dataset (six
underlying pairs, cycled to reach a requested count) are synthetic and
tiny, defined directly in `rlhf/data.py`. A public dataset such as
Anthropic's HH-RLHF or OpenAI's summarization comparisons would produce more
convincing final numbers, but would also take considerably longer to
download, tokenize, and train against, working against this project's other
goal of running the entire pipeline, all eight stages, end to end in a few
minutes on a laptop CPU with no GPU and no external downloads beyond the
model weights themselves. The cost of that choice is direct and reported
honestly in `docs/RESULTS.md`: the trained models overfit fast and the
aligned model's advantage over the base model does not reliably transfer to
prompts outside the training topics. Swapping in a larger, public preference
dataset would be a natural next step and requires no code changes beyond
what feeds `build_synthetic_preference_dataset`'s call sites in
`run_pipeline.py`.

## DPO, IPO, KTO, ORPO, and SimPO are compared on one batch, only DPO is trained

Actually training five separate policies, one per preference optimization
method, and comparing all five with their own win rate evaluation, was
considered and rejected for this iteration of the project on cost grounds:
it would mean five times the training and evaluation compute for a
comparison whose main value, showing that the loss formulas behave
differently by construction, does not require five trained checkpoints to
demonstrate. Instead, DPO is the one method whose weights are actually
updated, and IPO, KTO, ORPO, and SimPO are evaluated once, on the final
batch DPO trained against. That is enough to see a genuinely interesting,
correct result: IPO's loss on that batch is large (in the hundreds) not
because anything is broken, but because IPO is designed around a *fixed*
target preference margin (`1 / (2 * beta)`), while DPO has no such ceiling
and keeps rewarding the policy for separating chosen from rejected further
and further apart. Training DPO and then scoring that result under IPO's
loss is a direct, honest illustration of a real, documented difference
between the two methods. See `docs/RESULTS.md` for the numbers and a fuller
explanation. Training all five to convergence and comparing final win rates
is the natural extension if this project's scope grows.
