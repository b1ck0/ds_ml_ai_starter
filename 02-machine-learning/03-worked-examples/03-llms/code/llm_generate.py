"""SPEC-ML-11 worked example: prompting a small local instruction-tuned LLM.

Everything here runs on CPU, downloads once from the Hugging Face Hub, and produces REAL
generations -- nothing printed or written to an artefact is hand-written.

Model: HuggingFaceTB/SmolLM-135M-Instruct (Apache-2.0, 135M params, ~2048-token context),
per research/NOTE-ML-8-transformer-and-llm.md. It is the primary recommendation in that
NOTE and it loads and generates on CPU in this environment (confirmed below at runtime) --
no fallback to distilgpt2 was needed.

Five things happen here, in order:

1. Load the model + tokenizer, confirm its size and context window directly from the
   loaded config (not asserted from memory).
2. Zero-shot vs. few-shot prompting: the SAME sentiment-classification question, once with
   no examples and once with two worked examples given as prior chat turns. Both are real
   `model.generate()` calls.
3. The context window: count tokens with the tokenizer, show what "2048 tokens" costs in
   practice, and force real truncation on a prompt that is deliberately too long.
4. Decoding at the application level: the SAME prompt under five decoding settings
   (greedy + four sampling configurations), plus a determinism check -- same seed, same
   sampling settings, run twice, byte-for-byte identical output.
5. Limits: three questions this 135M-parameter model cannot possibly answer correctly
   (today's date, a 2026 sports result, the installed library's own version) to observe
   confident, fluent, WRONG output -- hallucination and knowledge cutoff, made concrete.

Environment: .venv-ml (Python 3.13, torch==2.14.0+cpu, transformers==5.16.1), CPU only.
Run: .venv-ml/Scripts/python.exe llm_generate.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Windows consoles default to a codepage (cp1252) that cannot print every BPE token
# string the tokenizer produces (e.g. the "Ġ" marker for a leading space) -- force
# UTF-8 stdout so Section 3's raw-token printout never crashes the run.
sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"  # Apache-2.0 -- NOTE-ML-8-transformer-and-llm.md
DEVICE = "cpu"
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"


def load_model() -> tuple:
    """Load SmolLM-135M-Instruct and print facts read from its OWN config, not memory."""
    print("=" * 78)
    print("1. Loading HuggingFaceTB/SmolLM-135M-Instruct")
    print("=" * 78)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    load_seconds = time.perf_counter() - t0

    num_params = model.num_parameters()
    ctx_window = model.config.max_position_embeddings
    print(f"load time:              {load_seconds:.1f}s (first run downloads the weights)")
    print(f"parameters:              {num_params:,}")
    print(f"architecture:            {model.config.architectures}")
    print(f"max_position_embeddings: {ctx_window}  (the trained context window, in tokens)")
    print(f"vocab_size:              {tokenizer.vocab_size:,}")
    print(f"has chat template:       {tokenizer.chat_template is not None}")
    print()
    return tokenizer, model


def chat_generate(
    tokenizer, model, messages: list[dict], max_new_tokens: int = 30, **gen_kwargs
) -> tuple[str, int, float]:
    """Apply the chat template, generate, and return (new_text_only, prompt_tokens, seconds).

    `set_seed` runs immediately before every call so a `do_sample=True` run reproduces
    exactly on a re-run; greedy (`do_sample=False`) is deterministic regardless of seed.
    """
    set_seed(SEED)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_tokens = inputs["input_ids"].shape[1]
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    elapsed = time.perf_counter() - start
    new_ids = output_ids[0][prompt_tokens:]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return new_text, prompt_tokens, elapsed


REVIEW = "The battery life is amazing and it charges fast."


def section_2_zero_shot_vs_few_shot(tokenizer, model) -> list[dict]:
    """Same classification question, zero-shot vs. few-shot (worked examples as chat turns)."""
    print("=" * 78)
    print("2. Zero-shot vs. few-shot prompting (sentiment classification)")
    print("=" * 78)

    question = (
        f'Classify the sentiment as Positive or Negative, respond with just the label: "{REVIEW}"'
    )

    zero_shot_messages = [{"role": "user", "content": question}]

    few_shot_messages = [
        {
            "role": "user",
            "content": 'Classify the sentiment as Positive or Negative, respond with just '
            'the label: "This is the worst purchase I ever made."',
        },
        {"role": "assistant", "content": "Negative"},
        {
            "role": "user",
            "content": 'Classify the sentiment as Positive or Negative, respond with just '
            'the label: "Works perfectly, very happy with it."',
        },
        {"role": "assistant", "content": "Positive"},
        {"role": "user", "content": question},
    ]

    rows: list[dict] = []
    for name, messages in [("zero_shot", zero_shot_messages), ("few_shot", few_shot_messages)]:
        text, prompt_tokens, elapsed = chat_generate(
            tokenizer, model, messages, max_new_tokens=25, do_sample=False
        )
        print(f"[{name}] prompt_tokens={prompt_tokens} ({elapsed:.2f}s)")
        print(f"  -> {text!r}\n")
        rows.append(
            {
                "setting": name,
                "num_chat_turns": len(messages),
                "prompt_tokens": prompt_tokens,
                "question": question,
                "generated_text": text,
                "seconds": round(elapsed, 2),
            }
        )
    return rows


def section_3_context_window(tokenizer, model) -> list[dict]:
    """Count tokens with the tokenizer, then force real truncation and a near-boundary run."""
    print("=" * 78)
    print("3. The context window -- counting tokens, hitting the ceiling")
    print("=" * 78)
    rows: list[dict] = []

    short_text = "The capital of France is"
    short_ids = tokenizer.encode(short_text)
    short_tokens = tokenizer.convert_ids_to_tokens(short_ids)
    print(f"text: {short_text!r}")
    print(f"token ids ({len(short_ids)}): {short_ids}")
    print(f"token strings: {short_tokens}")
    print(
        "(the leading '\\u0120' marker is this tokenizer's byte-level-BPE way of marking "
        "'a space came before this token' -- visible directly in the printed token strings, "
        "not every token maps to a whole word)\n"
    )
    rows.append({"description": "short prompt, raw token count", "token_count": len(short_ids)})

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": short_text}], tokenize=False, add_generation_prompt=True
    )
    chat_tokens = len(tokenizer.encode(chat_prompt))
    print(f"same text through the chat template costs {chat_tokens} tokens "
          f"(vs. {len(short_ids)} raw) -- the <|im_start|>/<|im_end|> role markers are not free.\n")
    rows.append({"description": "same prompt, through chat template", "token_count": chat_tokens})

    ctx_window = model.config.max_position_embeddings
    long_text = (
        "In the year 2024, researchers published a detailed study on the history of the "
        "transformer architecture and its many applications across natural language "
        "processing, computer vision, and reinforcement learning. "
    ) * 60
    raw_long_ids = tokenizer.encode(long_text)
    print(f"a {len(long_text)}-character prompt encodes to {len(raw_long_ids)} tokens -- "
          f"already over this model's {ctx_window}-token context window.\n")
    rows.append(
        {"description": "long prompt, untruncated", "token_count": len(raw_long_ids)}
    )

    truncated = tokenizer(long_text, truncation=True, max_length=ctx_window, return_tensors="pt")
    truncated_count = truncated["input_ids"].shape[1]
    print(f"tokenizer(..., truncation=True, max_length={ctx_window}) cuts it down to "
          f"{truncated_count} tokens -- the tail of the prompt is silently dropped.\n")
    rows.append(
        {
            "description": f"long prompt, truncated to max_length={ctx_window}",
            "token_count": truncated_count,
        }
    )

    max_new_tokens = 100
    budget = ctx_window - max_new_tokens
    budget_enc = tokenizer(long_text, truncation=True, max_length=budget, return_tensors="pt")
    budget_count = budget_enc["input_ids"].shape[1]
    print(f"reserving room for {max_new_tokens} generated tokens means the PROMPT budget is "
          f"only {ctx_window} - {max_new_tokens} = {budget} tokens: "
          f"tokenizer(..., max_length={budget}) -> {budget_count} tokens.\n")
    rows.append(
        {
            "description": f"prompt budget (ctx {ctx_window} - max_new_tokens {max_new_tokens})",
            "token_count": budget_count,
        }
    )

    # Push a real generate() call right up against the trained window and read the
    # library's own warning -- this is transformers 5.16.1's actual runtime behaviour,
    # not a hypothetical.
    at_limit_inputs = tokenizer(
        long_text, truncation=True, max_length=ctx_window, return_tensors="pt"
    ).to(DEVICE)
    print(f"generating 5 new tokens with a full {ctx_window}-token prompt "
          f"(this is slow -- attention cost grows with sequence length):")
    t0 = time.perf_counter()
    with torch.no_grad():
        near_limit_out = model.generate(
            **at_limit_inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    near_limit_elapsed = time.perf_counter() - t0
    print(f"  succeeded in {near_limit_elapsed:.1f}s, output length "
          f"{near_limit_out.shape[1]} tokens ({ctx_window} prompt + 5 new).\n")
    rows.append(
        {
            "description": f"generate() at the {ctx_window}-token boundary (+5 new tokens)",
            "token_count": int(near_limit_out.shape[1]),
        }
    )

    return rows


def section_4_decoding_app_level(tokenizer, model) -> None:
    """Same prompt, five decoding settings, plus a same-seed determinism check."""
    print("=" * 78)
    print("4. Decoding at the application level -- temperature and top-p")
    print("=" * 78)
    messages = [{"role": "user", "content": "Write one sentence about the ocean."}]

    settings: dict[str, dict] = {
        "greedy": dict(do_sample=False),
        "temperature_0.5": dict(do_sample=True, temperature=0.5, top_k=0, top_p=1.0),
        "temperature_1.0": dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0),
        "temperature_1.5": dict(do_sample=True, temperature=1.5, top_k=0, top_p=1.0),
        "top_p_0.9": dict(do_sample=True, temperature=1.0, top_k=0, top_p=0.9),
    }
    for name, kwargs in settings.items():
        text, _, elapsed = chat_generate(tokenizer, model, messages, max_new_tokens=30, **kwargs)
        print(f"[{name}] {kwargs} ({elapsed:.2f}s)")
        print(f"  -> {text!r}\n")

    print("Determinism check -- same sampling settings, same seed, run twice:")
    run1, _, _ = chat_generate(
        tokenizer, model, messages, max_new_tokens=30,
        do_sample=True, temperature=0.5, top_p=0.9,
    )
    run2, _, _ = chat_generate(
        tokenizer, model, messages, max_new_tokens=30,
        do_sample=True, temperature=0.5, top_p=0.9,
    )
    print(f"  run 1 -> {run1!r}")
    print(f"  run 2 -> {run2!r}")
    print(f"  identical: {run1 == run2}\n")


def section_5_limits(tokenizer, model) -> list[dict]:
    """Ask three questions this model cannot possibly answer correctly. Read the output."""
    print("=" * 78)
    print("5. Limits -- hallucination and knowledge cutoff, made concrete")
    print("=" * 78)
    questions = [
        "What is today's date?",
        "Who won the Ballon d'Or in 2026?",
        "What is the current version of the transformers Python library?",
    ]
    rows: list[dict] = []
    for q in questions:
        text, _, elapsed = chat_generate(
            tokenizer, model, [{"role": "user", "content": q}], max_new_tokens=40, do_sample=False
        )
        print(f"Q: {q}")
        print(f"A: {text!r} ({elapsed:.2f}s)\n")
        rows.append({"question": q, "answer": text})
    return rows


def write_zero_shot_few_shot_artefact(rows: list[dict]) -> Path:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTEFACTS_DIR / "zero_shot_vs_few_shot.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "setting", "num_chat_turns", "prompt_tokens", "question",
                "generated_text", "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote artefact: {csv_path}")
    return csv_path


def write_context_window_artefact(rows: list[dict]) -> Path:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTEFACTS_DIR / "context_window.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["description", "token_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote artefact: {csv_path}")
    return csv_path


def main() -> None:
    tokenizer, model = load_model()
    zero_few_rows = section_2_zero_shot_vs_few_shot(tokenizer, model)
    context_rows = section_3_context_window(tokenizer, model)
    section_4_decoding_app_level(tokenizer, model)
    section_5_limits(tokenizer, model)

    write_zero_shot_few_shot_artefact(zero_few_rows)
    write_context_window_artefact(context_rows)


if __name__ == "__main__":
    main()
