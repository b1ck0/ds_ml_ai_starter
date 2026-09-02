# SPEC-ML-11: LLMs in Practice — generation, prompting, and limits

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Worked Examples (LLMs)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-9, SPEC-ML-10
**Nature:** RUNNABLE with a small local model; hosted-API usage shown as grounded reference.

## Intent
Bridge from "a transformer block" to "an LLM you use". Show a small local causal LM generating and
following simple instructions, explain context windows, and honestly frame capabilities/limits —
setting up the Agentic Engineering subject (RAG, tools, MCP).

## Learning objectives
- LO1 — Run a small local instruction-capable LM and prompt it; observe few-shot vs zero-shot.
- LO2 — Explain the context window (tokens in/out), why it's finite, and how it bounds prompting/RAG.
- LO3 — Explain temperature/top-p at the application level and reproducibility (seeding, determinism limits).
- LO4 — Frame LLM limits (hallucination, knowledge cutoff, no grounding) → why RAG/tools/MCP exist (forward-link to Agentic).

## Scope
In: small local LM generation + prompting patterns, context-window explanation, decoding at app level, limits.
Out: training/fine-tuning LLMs, hosted-provider SDK depth (reference only; the Agentic subject uses providers).

## Outline
1. What & why — from block to model to assistant; the prompt as the program.
2. Run a small local instruct model; zero-shot vs few-shot examples.
3. Context window — count tokens; show truncation; why long context matters.
4. Decoding at the app level — temperature/top-p; determinism caveats.
5. Limits → the bridge to Agentic — hallucination/cutoff motivate RAG, tools, MCP.
6. Pitfalls — over-trusting output, prompt injection preview, token budget blowout.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/llms/llm-text-generation.md"
- Code: "Machine Learning/Worked Examples/llms/code/llm_generate.py"
- Artefacts: zero-shot vs few-shot output comparison; a token-count/context illustration.

## Claims to ground (Haiku, before writing)
- [ ] Recommend a SMALL CPU-runnable instruction-tuned model id that downloads freely (e.g. a small Qwen/TinyLlama/SmolLM instruct — verify size, licence, and that it runs on CPU) OR fall back to distilgpt2 with the limits noted. Verify the transformers API to load+generate with a chat template.
- [ ] Verify how to count tokens with the model's tokenizer and typical context-window sizes for the chosen model.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — llm_generate.py RUNS on CPU with a real small model and produces real generations + token counts; snippet-check passes; no fabricated outputs. If no suitable small model runs in-sandbox, ESCALATE. AC3 — model id/licence + tokenizer/context facts grounded. AC4 — the limits→RAG/tools bridge lands; sets up the Agentic subject.

## Gates
Entry: approved; notes landed. Exit: DoD checklist. Uses .venv-ml (+ transformers).
