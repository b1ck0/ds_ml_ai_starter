# SPEC-ML-10: The Transformer from the Inside

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Worked Examples (LLMs)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-2, SPEC-ML-3
**Nature:** RUNNABLE — build the pieces in PyTorch on tiny tensors (CPU).

## Intent
Open the black box. Implement scaled dot-product attention and a single transformer block in PyTorch
on tiny inputs so the reader can SEE the shapes and the attention matrix — the foundation under every
LLM.

## Learning objectives
- LO1 — Implement scaled dot-product attention (Q,K,V) and explain each step and the √d scaling.
- LO2 — Extend to multi-head attention and explain what heads buy you.
- LO3 — Assemble one transformer block (attention + residual + layernorm + feed-forward) and run a tensor through it.
- LO4 — Explain positional encoding and causal masking, and visualise an attention matrix.

## Scope
In: attention + multi-head + one encoder block, positional encoding, causal mask, attention-matrix viz — all on tiny tensors.
Out: full training, full model stacks (use HuggingFace for that in ML-9/ML-11).

## Outline
1. What & why — attention as content-based lookup; why it beat recurrence.
2. Scaled dot-product attention — implement + inspect the attention weights.
3. Multi-head attention — split/concat heads.
4. A transformer block — residual + layernorm + FFN; run a tensor through; check shapes.
5. Positional encoding + causal mask — why order and masking matter; visualise attention.
6. Pitfalls — shape/transpose bugs, forgetting the scale, mask direction.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/llms/transformer-internals.md"
- Code: "Machine Learning/Worked Examples/llms/code/transformer_from_scratch.py"
- Artefacts: an attention-matrix heatmap; a shapes-through-the-block table.

## Claims to ground (Haiku, before writing)
- [ ] Verify the scaled dot-product + multi-head attention formulas and the transformer block structure against "Attention Is All You Need" / an authoritative reference.
- [ ] Confirm the torch APIs used (nn.Linear, softmax, layernorm) on the installed torch version, OR that a pure-numpy version is viable (prefer torch for authenticity).

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — transformer_from_scratch.py RUNS on CPU, produces the attention heatmap + shapes table; snippet-check passes. AC3 — attention/block formulas grounded with citation. AC4 — every shape shown; a Java dev can follow the tensor flow.

## Gates
Entry: approved; notes landed. Exit: DoD checklist. Uses .venv-ml.
