# SPEC-ML-9: Text Generation with a Decoder Model

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Worked Examples (Natural Language)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-8
**Nature:** RUNNABLE ON CPU with a SMALL decoder model.

## Intent
Text generation, done right. IMPORTANT CORRECTION to the original curriculum note: RoBERTa is an
ENCODER-only model and cannot generate text. Generation needs a decoder (GPT-family, e.g.
distilgpt2/gpt2) or an encoder–decoder (T5). Teach autoregressive generation and decoding strategies
on a small CPU-friendly model.

## Learning objectives
- LO1 — Explain why generation requires a decoder (causal LM) and how autoregressive next-token prediction works.
- LO2 — Generate text with a small model via HuggingFace `generate`, and explain the decoding knobs: greedy, beam, temperature, top-k, top-p.
- LO3 — Show how decoding parameters change output (deterministic vs diverse) with real examples.
- LO4 — Contrast decoder (GPT) vs encoder (BERT/RoBERTa) vs encoder–decoder (T5) and their jobs.

## Scope
In: small decoder LM generation on CPU, decoding strategies with real outputs, model-family contrast.
Out: training/fine-tuning an LM, RLHF (mention), large-model serving (→ Agentic/cloud).

## Outline
1. What & why — encoder vs decoder vs seq2seq; why RoBERTa can't generate (correct the myth).
2. Autoregressive generation — next-token loop, the tokenizer round-trip.
3. Decoding strategies — greedy/beam/temperature/top-k/top-p; show the SAME prompt under each.
4. Model families recap — pick the right one for the job.
5. Pitfalls — repetition, hallucination, context length, pad/eos tokens.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/natural-language/text-generation.md"
- Code: "Machine Learning/Worked Examples/natural-language/code/text_generation.py"
- Artefacts: a table of the same prompt generated under different decoding settings (real outputs).

## Claims to ground (Haiku, before writing)
- [ ] Confirm RoBERTa is encoder-only (cannot do causal generation) from authoritative docs — cite it.
- [ ] Verify a SMALL CPU-friendly decoder model id that downloads freely (distilgpt2 / gpt2) + licence, and the current `transformers` generate() API + decoding params (do_sample, temperature, top_k, top_p, num_beams).

## Acceptance criteria
- [ ] AC1 — LOs delivered incl. the encoder-vs-decoder correction. AC2 — text_generation.py RUNS on CPU, generates REAL text under several decoding settings; snippet-check passes; no fabricated outputs. AC3 — RoBERTa-encoder fact + model id + generate API grounded. AC4 — decoding knobs shown to matter with real examples.

## Gates
Entry: approved; notes landed. Exit: DoD checklist. Uses .venv-ml (+ transformers).
