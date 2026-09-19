---
title: Fine-tuning, LoRA and RLHF
subject: ai
topic: fine-tuning
difficulty: advanced
year: 2026
source: In-house study note (after Hu et al. 2021; Ouyang et al. 2022)
---

# Fine-tuning: Full, LoRA, and RLHF

Pre-training gives a model general language ability; **fine-tuning** adapts it to a task,
style, or behaviour by continuing training on a smaller, targeted dataset.

## Full fine-tuning vs parameter-efficient methods

Full fine-tuning updates all weights — for a 70B-parameter model that means hundreds of
gigabytes of optimizer state and a separate full copy per task. **Parameter-efficient
fine-tuning (PEFT)** freezes the base model and trains a tiny number of new parameters.

## LoRA — Low-Rank Adaptation (Hu et al., 2021)

For beginners: instead of rebuilding a whole piano to change its sound, LoRA slips a thin
felt strip over the strings — a small, removable modification that changes behaviour.

Technically: a fine-tuning update to a weight matrix $W \in \mathbb{R}^{d \times k}$ is
$W' = W + \Delta W$. LoRA assumes $\Delta W$ has low intrinsic rank and factorises it:

$$W' = W + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in
\mathbb{R}^{r \times k},\; r \ll \min(d, k)$$

Only $A$ and $B$ are trained (often <1% of parameters). $W$ stays frozen, so a LoRA
"adapter" is a few megabytes you can swap per task or merge into $W$ for zero inference
overhead. **QLoRA** goes further: quantise the frozen base model to 4-bit and train LoRA
adapters on top — fine-tuning a 65B model on a single GPU.

## RLHF — Reinforcement Learning from Human Feedback (Ouyang et al., 2022)

Pre-trained models predict text; they don't inherently prefer helpful, honest, harmless
answers. RLHF aligns them with human preferences in three stages:

1. **SFT:** supervised fine-tuning on human-written demonstrations of good responses.
2. **Reward model:** humans rank pairs of model responses; a separate model is trained to
   predict these preferences, yielding a scalar reward $r(x, y)$.
3. **RL optimisation:** the policy (the LLM) is optimised — classically with PPO — to
   maximise reward, with a KL penalty $\beta \, \mathrm{KL}(\pi \,\|\, \pi_{\text{SFT}})$
   keeping it close to the SFT model so it doesn't degenerate into reward hacking.

**DPO (Direct Preference Optimization)** collapses stages 2–3 into a single supervised loss
on preference pairs — no explicit reward model, no RL loop — and has become a popular,
stabler alternative. **RLAIF / Constitutional AI** replaces some human labelling with
AI feedback guided by written principles.

## Choosing an adaptation strategy

- Need new *facts* that change often → RAG, not fine-tuning.
- Need a *style, format, or skill* (JSON output, domain tone, a new language) → SFT/LoRA.
- Need *preference alignment* at scale → RLHF/DPO (usually the base-model vendor's job).
- Prompt engineering first, always: it's the cheapest experiment and often sufficient.
