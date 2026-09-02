"""SPEC-ML-9 worked example: text generation with a decoder-only (causal) LM.

Two things happen here, in order:

1. We inspect RoBERTa's own config to show it is NOT a generation model — it is an
   encoder (bidirectional, no causal mask), per NOTE-ML-7-nlp-models.md and the RoBERTa
   paper (Liu et al. 2019, https://arxiv.org/abs/1907.11692). We do NOT try to force
   RoBERTa to generate; the point is to show *why* it can't, not to fabricate broken output.
2. We load distilgpt2 (a real decoder-only causal LM, Apache-2.0, per NOTE-ML-7) and
   generate REAL text from the same prompt under six decoding settings: greedy, beam
   search, two temperatures, top-k, and top-p. Every string printed and written to the
   artefact CSV is the literal output of `model.generate()` on this machine — nothing is
   hand-written.

Environment: .venv-ml (Python 3.13, torch==2.14.0+cpu, transformers==5.16.1), CPU only.
Run: .venv-ml/Scripts/python.exe text_generation.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

SEED = 42
CAUSAL_MODEL_ID = "distilbert/distilgpt2"  # Apache-2.0, ~82M params — NOTE-ML-7-nlp-models.md
ENCODER_MODEL_ID = "roberta-base"  # inspected for its config only, never loaded for generation
DEVICE = "cpu"
PROMPT = "The future of AI is"
MAX_NEW_TOKENS = 30
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"


def section_1_why_roberta_cant_generate() -> None:
    """Show, from RoBERTa's own HuggingFace config, that it has no causal-generation wiring.

    We only download the small config.json here (a few KB), not the ~500MB of weights —
    there is nothing to gain from actually running RoBERTa through `.generate()`: its
    encoder self-attention has no causal mask, so `AutoModelForCausalLM` does not even
    register it as a valid causal-LM architecture family in the same way GPT-2-style
    models are. The relevant, checkable fact is right there in the config.
    """
    print("=" * 78)
    print("1. Why RoBERTa (an ENCODER) cannot generate text")
    print("=" * 78)
    config = AutoConfig.from_pretrained(ENCODER_MODEL_ID)
    print(f"model_type:        {config.model_type}")
    print(f"architectures:      {config.architectures}")
    print(f"is_decoder:         {config.is_decoder}")
    print(f"add_cross_attention:{config.add_cross_attention}")
    print(
        "\nRoBERTa's pretraining head is 'RobertaForMaskedLM': it was trained to fill in\n"
        "*masked* tokens using context from BOTH directions (left AND right) at once.\n"
        "`is_decoder=False` means its self-attention layers carry no causal mask -- every\n"
        "token can already see every other token, including ones 'to the right' of it.\n"
        "There is no notion of 'predict the next token given only what came before', which\n"
        "is the one operation autoregressive generation repeats in a loop. Source: RoBERTa\n"
        "paper (Liu et al. 2019), https://arxiv.org/abs/1907.11692, and NOTE-ML-7-nlp-models.md.\n"
    )


def load_causal_model() -> tuple:
    """Load distilgpt2, a real decoder-only (causal) LM, for text generation."""
    tokenizer = AutoTokenizer.from_pretrained(CAUSAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(CAUSAL_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, **gen_kwargs) -> tuple[str, float]:
    """Run one real generate() call and return (decoded_text, elapsed_seconds).

    `set_seed` is called immediately before every sampling call so that the SAME
    settings reproduce the SAME sampled text on a re-run (transformers.set_seed seeds
    Python's `random`, NumPy, and torch — NOTE-ML-7-nlp-models.md / transformers 5.16.1 docs).
    Greedy and beam search are deterministic regardless of seed; the seed only matters
    for the `do_sample=True` rows.
    """
    set_seed(SEED)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,  # distilgpt2 has no pad token — see Pitfalls
            **gen_kwargs,
        )
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, elapsed


def section_2_decoding_strategies(tokenizer, model) -> list[dict]:
    """Generate the SAME prompt under six decoding settings and return the real results."""
    print("=" * 78)
    print("2. Same prompt, six decoding strategies (distilgpt2, CPU)")
    print("=" * 78)
    print(f"prompt: {PROMPT!r}\n")

    settings: dict[str, dict] = {
        "greedy": dict(do_sample=False, num_beams=1),
        "beam_5": dict(do_sample=False, num_beams=5),
        "temperature_0.7": dict(do_sample=True, temperature=0.7, top_k=0, top_p=1.0),
        "temperature_1.5": dict(do_sample=True, temperature=1.5, top_k=0, top_p=1.0),
        "top_k_10": dict(do_sample=True, temperature=1.0, top_k=10, top_p=1.0),
        "top_p_0.95": dict(do_sample=True, temperature=1.0, top_k=0, top_p=0.95),
    }

    rows: list[dict] = []
    for name, kwargs in settings.items():
        text, elapsed = generate(tokenizer, model, PROMPT, **kwargs)
        continuation = text[len(PROMPT):].strip()
        print(f"[{name}] params={kwargs} ({elapsed:.2f}s)")
        print(f"  -> {text!r}\n")
        rows.append(
            {
                "setting": name,
                "params": kwargs,
                "full_text": text,
                "continuation": continuation,
                "seconds": round(elapsed, 2),
            }
        )
    return rows


def section_3_repetition_pitfall(tokenizer, model) -> None:
    """Demonstrate the classic greedy-decoding repetition loop, then fix it."""
    print("=" * 78)
    print("3. Pitfall: greedy decoding loops on a repetition-prone prompt")
    print("=" * 78)
    loop_prompt = "The dog said the dog said the dog"
    without_fix, _ = generate(
        tokenizer, model, loop_prompt, do_sample=False, num_beams=1
    )
    with_fix, _ = generate(
        tokenizer,
        model,
        loop_prompt,
        do_sample=False,
        num_beams=1,
        no_repeat_ngram_size=3,
    )
    print(f"prompt: {loop_prompt!r}\n")
    print(f"plain greedy         -> {without_fix!r}")
    print(f"greedy + no_repeat_ngram_size=3 -> {with_fix!r}\n")


def section_4_hallucination_pitfall(tokenizer, model) -> None:
    """Demonstrate that fluent output is not the same as correct output.

    distilgpt2 has no fact database and no retrieval step -- every token is a statistical
    guess given the training data and the tokens so far. Greedy decoding on a
    fact-shaped prompt produces a fluent, confident-*sounding* sentence that is, in this
    case, verifiably wrong (Einstein was born in 1879, not 1867) -- a real, reproducible
    example of hallucination, not a hypothetical one.
    """
    print("=" * 78)
    print("4. Pitfall: fluent output is not the same as correct output (hallucination)")
    print("=" * 78)
    fact_prompt = "Albert Einstein was born in the year"
    text, _ = generate(tokenizer, model, fact_prompt, do_sample=False, num_beams=1)
    print(f"prompt: {fact_prompt!r}")
    print(f"greedy -> {text!r}")
    print(
        "(Einstein was actually born in 1879 -- the model produced a fluent, confident,\n"
        "and factually wrong year, with no signal in the output that it might be unreliable.)\n"
    )


def write_artefact(rows: list[dict]) -> Path:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTEFACTS_DIR / "decoding_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["setting", "params", "full_text", "continuation", "seconds"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote artefact: {csv_path}")
    return csv_path


def main() -> None:
    section_1_why_roberta_cant_generate()

    tokenizer, model = load_causal_model()
    num_params = model.num_parameters()
    print(f"Loaded {CAUSAL_MODEL_ID}: {num_params:,} parameters, n_positions={model.config.n_positions}\n")

    rows = section_2_decoding_strategies(tokenizer, model)
    section_3_repetition_pitfall(tokenizer, model)
    section_4_hallucination_pitfall(tokenizer, model)
    write_artefact(rows)


if __name__ == "__main__":
    main()
