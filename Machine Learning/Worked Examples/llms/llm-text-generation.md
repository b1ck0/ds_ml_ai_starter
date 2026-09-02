# LLMs in Practice — Generation, Prompting, and Limits

*Machine Learning · Worked Examples · LLMs · SPEC-ML-11*

Every service you've ever written has a fixed interface: a method signature, a REST endpoint schema,
a message contract. You change behavior by changing code and redeploying. An instruction-tuned LLM
flips that around — the "interface" is natural language, and you change behavior by changing the
**text you send it**, with no redeploy at all. [transformer-internals.md](transformer-internals.md)
(SPEC-ML-10) built the mechanism — attention, residuals, LayerNorm, a stack of transformer blocks —
from scratch, on tensors small enough to print. [text-generation.md](../natural-language/text-generation.md)
(SPEC-ML-9) showed the next-token loop that turns those blocks into a text generator, using
`distilgpt2`, a model that was never taught to follow instructions. This chapter closes the gap: a
real, small, **instruction-tuned** model, prompted the way you'd actually use one — and an honest
look at exactly where that stops working, which is where the next subject, Agentic Engineering,
picks up.

The mental model worth carrying through this whole chapter: **the prompt is the program.** Not a
metaphor — literally the only input you control. There's no separate "configuration" layer, no
method overload, no compiled contract the runtime enforces for you. Get the prompt wrong and the
model doesn't throw an exception; it just generates something else, fluently and without complaint.
That's the central engineering challenge this chapter sets up, and the one Agentic Engineering spends
a whole subject solving.

Everything below runs from [`llm_generate.py`](code/llm_generate.py), and every claim about the
model, its license, its context window, or its tokenizer API traces to
[research/NOTE-ML-8-transformer-and-llm.md](../../../research/NOTE-ML-8-transformer-and-llm.md),
which cites the model's own Hugging Face card:
[source: HuggingFaceTB/SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)
(checked 2026-09-02).

### Environment

```text
torch==2.14.0+cpu
transformers==5.16.1
Python 3.13 (.venv-ml)
```

Installed and verified live in the project's shared `.venv-ml` virtual environment (checked
2026-09-02), matching NOTE-ML-8-transformer-and-llm.md. Everything in this chapter runs on CPU. The
model is `HuggingFaceTB/SmolLM-135M-Instruct` — the NOTE's primary recommendation (Apache-2.0
license, 135M parameters) — and it loaded and generated on CPU without issue in this environment (2.3s
load time, weights already warm in the local Hugging Face cache from earlier verification runs; a
genuinely cold first run downloads roughly 300–400MB of weights, per the NOTE). No fallback to
`distilgpt2` was needed.

## 1. What & why — from block to model to assistant

SPEC-ML-9 already drew the line between an **encoder** (understands, doesn't generate — RoBERTa) and
a **decoder** (generates, one token at a time — `distilgpt2`, and every GPT-style model). SmolLM-135M-Instruct
is a decoder too — same next-token loop, same `generate()` call. What makes it different from
`distilgpt2` is not its architecture, it's what happened to it *after* pretraining: it was fine-tuned
on instruction-following conversation data (UltraChat, HelpSteer, per NOTE-ML-8-transformer-and-llm.md)
so that, given something shaped like a chat turn, it tends to produce something shaped like a helpful
reply, rather than just "the statistically likely next words."

That distinction is the whole reason `distilgpt2` couldn't reliably follow an instruction in SPEC-ML-9
and this model can attempt to: **instruction-tuning is a second training phase that teaches a model
the *shape* of a request/response exchange, on top of a first phase that only taught it to predict
the next token of raw text.** Nothing about the transformer block itself changed — same attention,
same feed-forward layers, same causal mask from SPEC-ML-10. What changed is the data used to nudge
the weights afterward.

To use that shape, the model doesn't take a raw string — it takes a **chat template**: a small
piece of the tokenizer's own configuration that knows how this specific model was trained to see a
conversation, with special role-marker tokens (`<|im_start|>`, `<|im_end|>`) wrapping each turn
(`tokenizer.apply_chat_template`, confirmed against transformers 5.16.1 in
NOTE-ML-8-transformer-and-llm.md). Think of it as a wire format — the same reason your Java service
doesn't hand-build JSON with string concatenation, you don't hand-build the prompt string either; the
tokenizer's chat template is the (de)serializer, and different model families use genuinely different
formats. Feed a model text formatted for a different model's chat template and you get plausible-looking
garbage, because it was never trained to parse that shape.

## 2. Running the model — zero-shot vs. few-shot

[`llm_generate.py`](code/llm_generate.py)'s `load_model()` loads the tokenizer and model once and
reads its size and context window straight from the loaded config — not asserted from memory:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
model.eval()

print(model.num_parameters())
print(model.config.max_position_embeddings)
print(tokenizer.chat_template is not None)
```

Actual output from the gate run:

```text
load time:              2.3s (first run downloads the weights)
parameters:              134,515,008
architecture:            ['LlamaForCausalLM']
max_position_embeddings: 2048  (the trained context window, in tokens)
vocab_size:              49,152
has chat template:       True
```

Two things worth noting before generating anything: **`architecture: ['LlamaForCausalLM']`** — SmolLM
is a small model trained with the same architecture family as Meta's Llama models (decoder-only,
rotary position embeddings instead of SPEC-ML-10's sinusoidal ones — the underlying attention
mechanism is unchanged); and **`max_position_embeddings: 2048`** is the context window this whole
chapter builds toward in Section 3.

### Zero-shot vs. few-shot, same question

"Zero-shot" means you ask for the task with no worked examples — you trust the instruction-tuning
alone to produce the right shape of answer. "Few-shot" means you show the model a couple of solved
examples first, as prior turns in the same conversation, before asking your real question — you're
using the model's in-context learning (attending back over the earlier turns, per SPEC-ML-10's
attention mechanism) to demonstrate the expected format directly, rather than only describing it.

`section_2_zero_shot_vs_few_shot()` asks the identical sentiment-classification question both ways.
Zero-shot is one chat turn:

```python
question = (
    'Classify the sentiment as Positive or Negative, respond with just the label: '
    '"The battery life is amazing and it charges fast."'
)
zero_shot_messages = [{"role": "user", "content": question}]
```

Few-shot prepends two solved examples as real prior `user`/`assistant` turns, then asks the same
question:

```python
few_shot_messages = [
    {"role": "user", "content": 'Classify the sentiment as Positive or Negative, respond with '
                                 'just the label: "This is the worst purchase I ever made."'},
    {"role": "assistant", "content": "Negative"},
    {"role": "user", "content": 'Classify the sentiment as Positive or Negative, respond with '
                                 'just the label: "Works perfectly, very happy with it."'},
    {"role": "assistant", "content": "Positive"},
    {"role": "user", "content": question},
]
```

Both run through `model.generate()` with greedy decoding (deterministic — Section 4 covers why).
Actual output from the gate run, and the same data in
[`artefacts/zero_shot_vs_few_shot.csv`](artefacts/zero_shot_vs_few_shot.csv):

| Setting | Chat turns | Prompt tokens | Generated text |
|---|---|---|---|
| `zero_shot` | 1 | 35 | *"Here's a possible classification of the sentiment as positive or negative:\n\n**Positive Sentiment:**\n\n\* "The"* |
| `few_shot` | 5 | 108 | *"Positive: "The battery life is amazing and it charges fast."\n\nNegative: "The battery life is amazing and it"* |

Read this honestly, not generously: **neither run obeys "respond with just the label."** This is a
135M-parameter model — two orders of magnitude smaller than the models people mean when they say "the
LLM" in casual conversation — and instruction-following is a capability that scales with model size.
But the *difference between the two rows is real and worth the extra tokens*: zero-shot opens with a
meta-commentary preamble ("Here's a possible classification...") before it even mentions the review;
few-shot's very first token is `Positive` — the correct label, immediately, in the position the
worked examples showed an answer belongs. The few-shot examples didn't fix the model's tendency to
keep generating past the answer (it goes on to hallucinate a second, contradictory `Negative:` line —
more on this pattern in Section 5), but they visibly pulled the *start* of the response toward the
demonstrated shape. That's in-context learning working, partially, at a scale where "partially" is
the honest word.

## 3. The context window — a finite token budget, not a soft guideline

Every LLM has a maximum number of tokens it can attend over in a single forward pass — prompt tokens
plus generated tokens combined. For SmolLM-135M-Instruct that ceiling is `max_position_embeddings =
2048`, read directly from the loaded config above and confirmed against the model card in
NOTE-ML-8-transformer-and-llm.md. This is not a rate limit or a pricing tier — it's an architectural
fact: the model's attention computation (SPEC-ML-10, Section 2) has shapes sized for sequences up to
this length, and the model was never trained on anything longer.

### Text isn't tokens — count them

A token is not a word and not a character; it's whatever unit the tokenizer's vocabulary happens to
carve text into (SPEC-ML-9 already used this via `distilgpt2`'s tokenizer; SmolLM's is a different,
49,152-entry vocabulary). Count tokens directly with `tokenizer.encode()`:

```python
text = "The capital of France is"
ids = tokenizer.encode(text)
print(len(ids), ids)
print(tokenizer.convert_ids_to_tokens(ids))
```

Actual output:

```text
token ids (5): [504, 3575, 282, 4649, 314]
token strings: ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
```

Five words, five tokens here — but that's a coincidence of this particular short sentence, not a
rule. The leading `Ġ` marker in four of the five token strings is this tokenizer's byte-level BPE way
of encoding "a space preceded this token" as part of the token itself, visible directly in the printed
strings above; a longer or less common word can split into multiple sub-word tokens with no spaces at
all. Wrapping the same five words in the chat template costs more, not the same:

```text
same text through the chat template costs 14 tokens (vs. 5 raw)
```

The `<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n` scaffolding around every turn is
real token cost, paid on every single request — a system prompt and conversation history in a
multi-turn chat application are not free just because they "aren't the question."

### Real truncation, not a hypothetical

`section_3_context_window()` builds a prompt deliberately too long — the same paragraph repeated 60
times — and encodes it:

```python
long_text = ("In the year 2024, researchers published a detailed study on the history of the "
             "transformer architecture and its many applications across natural language "
             "processing, computer vision, and reinforcement learning. ") * 60
ids = tokenizer.encode(long_text)
```

Actual output:

```text
a 12660-character prompt encodes to 2221 tokens -- already over this model's 2048-token context window.
```

That real transformers 5.16.1 warning fires the moment the untruncated encode happens (from the
gate run, verbatim):

```text
[transformers] Token indices sequence length is longer than the specified maximum sequence length
for this model (2221 > 2048). Running this sequence through the model will result in indexing errors
```

Passing `truncation=True` forces the tokenizer to actually cut it down, dropping whatever doesn't
fit — silently, from the tail of the input:

```python
truncated = tokenizer(long_text, truncation=True, max_length=2048, return_tensors="pt")
```

```text
tokenizer(..., truncation=True, max_length=2048) cuts it down to 2048 tokens -- the tail of the
prompt is silently dropped.
```

**2048 is not the budget available to your prompt** — it's the budget for prompt *and* generated
output combined. Reserve room for the reply first:

```text
reserving room for 100 generated tokens means the PROMPT budget is only 2048 - 100 = 1948 tokens
```

The same data, gathered end to end, is in
[`artefacts/context_window.csv`](artefacts/context_window.csv):

| Description | Token count |
|---|---|
| Short prompt, raw | 5 |
| Same prompt, through chat template | 14 |
| Long prompt, untruncated | 2221 |
| Long prompt, truncated to `max_length=2048` | 2048 |
| Prompt budget (2048 context − 100 reserved for output) | 1948 |
| `generate()` at the 2048-token boundary (+5 new tokens) | 2053 |

### What actually happens if you don't truncate

The last row is the interesting one. `llm_generate.py` runs a real `generate()` call with a
full 2048-token prompt and asks for 5 more tokens — one token past the model's trained window:

```python
at_limit_inputs = tokenizer(long_text, truncation=True, max_length=2048, return_tensors="pt")
output = model.generate(**at_limit_inputs, max_new_tokens=5, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
```

Actual output, including a second real library warning:

```text
[transformers] This is a friendly reminder - the current text generation call has exceeded the
model's predefined maximum length (2048). Depending on the model, you may observe exceptions,
performance degradation, or nothing at all.
  succeeded in 14.5s, output length 2053 tokens (2048 prompt + 5 new).
```

It **did not raise**. This model's own config (Section 2) shows `architecture: ['LlamaForCausalLM']`
using rotary position embeddings, not the learned, fixed-size position-embedding table older GPT-2-style
models use (SPEC-ML-9's `n_positions=1024` pitfall) — so there's no hard index-out-of-bounds lookup to
crash on here. The library's own warning is explicit that this is model-dependent: "exceptions,
performance degradation, or nothing at all." That's a strictly worse failure mode for you as the
engineer than a clean exception: no stack trace to catch, just a model silently reasoning over tokens
it was never trained to attend across, with no guarantee the output degrades gracefully. **Never rely
on "it didn't crash" as evidence a prompt fit — check `len(tokenizer.encode(prompt))` against the
context window yourself, every time**, exactly as this section did before ever calling `generate()`.

This 2048-token ceiling is also the first concrete reason RAG (retrieval-augmented generation) exists,
previewed properly in Section 5: you cannot just paste an entire document, or an entire codebase, or a
full chat history into every prompt once it's larger than a few thousand tokens. Something has to
decide what's actually relevant enough to spend the budget on.

## 4. Decoding at the application level — temperature and top-p

SPEC-ML-9 covered decoding strategies in depth (greedy, beam search, temperature, top-k, top-p) using
`distilgpt2`. The mechanics are identical here — same `generate()` keyword arguments, confirmed
against transformers 5.16.1 in NOTE-ML-8-transformer-and-llm.md — this section's job is narrower:
show what those knobs look like from the **application** side, once you're calling an instruction-tuned
model through a prompt rather than hand-rolling continuations, and be precise about what "reproducible"
actually guarantees.

`section_4_decoding_app_level()` sends the same prompt, `"Write one sentence about the ocean."`,
through five decoding settings:

```python
settings = {
    "greedy": dict(do_sample=False),
    "temperature_0.5": dict(do_sample=True, temperature=0.5, top_k=0, top_p=1.0),
    "temperature_1.0": dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0),
    "temperature_1.5": dict(do_sample=True, temperature=1.5, top_k=0, top_p=1.0),
    "top_p_0.9": dict(do_sample=True, temperature=1.0, top_k=0, top_p=0.9),
}
```

Actual output from the gate run:

| Setting | Output |
|---|---|
| `greedy` | "The ocean is a vast and mysterious world, hidden from the prying eyes of humans. It is a vast, blue expanse that covers over 7" |
| `temperature_0.5` | "The ocean is a vast, mysterious, and awe-inspiring world that covers over 70% of our planet. It is a vast and" |
| `temperature_1.0` | "The ocean is a vast, blue expanse of water that surrounds the Earth, supporting life and shaping the planet's landscape. This vast blue world is home" |
| `temperature_1.5` | "The ocean is one of the most massive creations of humans on this planet, supporting immense ecosystems and giving life to countless creatures. It is a vast and" |
| `top_p_0.9` | "The ocean is a vast, blue expanse of water that covers about 71% of our planet, stretching from the Arctic to the South Pole and" |

Compared with SPEC-ML-9's `distilgpt2` sweep (which visibly degraded into incoherent word salad at
`temperature=1.5`), every row here stays grammatical, even at the highest temperature. That's a real,
measured difference — not because temperature works differently on this model, but because
instruction-tuning on conversational data pushes a model's output distribution toward "coherent
sentence" more strongly than raw next-token pretraining alone does. Higher temperature still means
more variation, sampled here from a distribution that was shaped to stay closer to sensible English in
the first place.

### Reproducibility — what the seed actually guarantees

`chat_generate()` calls `transformers.set_seed(42)` immediately before every `generate()` call. Greedy
decoding needs no seed — it's deterministic by construction, always picking the single highest-probability
token. Sampling (`do_sample=True`) is where the seed matters: it fixes the random draws used to turn a
probability distribution into a chosen token. The gate run calls the exact same sampling settings
twice in a row:

```python
run1, _, _ = chat_generate(tokenizer, model, messages, max_new_tokens=30,
                            do_sample=True, temperature=0.5, top_p=0.9)
run2, _, _ = chat_generate(tokenizer, model, messages, max_new_tokens=30,
                            do_sample=True, temperature=0.5, top_p=0.9)
```

Actual output:

```text
run 1 -> 'The ocean is a vast, mysterious, and awe-inspiring world that covers over 70% of our planet. It is a vast,'
run 2 -> 'The ocean is a vast, mysterious, and awe-inspiring world that covers over 70% of our planet. It is a vast,'
identical: True
```

Byte-for-byte identical, on this machine, this exact model checkpoint, this exact library version,
run in CPU-only mode. **That's the actual scope of the guarantee** — not "this prompt always produces
this text." Change the hardware (CPU vs. GPU, or a different CPU's floating-point rounding), the
batch size, or the `torch`/`transformers` version, and floating-point operations can accumulate in a
different order, which can flip a probability comparison right at a tie and cascade into a different
sampled token from there on. A seeded, sampled LLM call is reproducible the way a `HashMap`'s
iteration order is "deterministic" — true within one fixed environment, not a portable guarantee you
can rely on across machines or upgrades. If you need a hosted-provider API's output to reproduce
exactly, most providers document a comparable seed parameter with the same "best effort, not a
contract" caveat — worth reading their docs before depending on it in a test suite.

## 5. Limits — hallucination, knowledge cutoff, and why RAG/tools/MCP exist

Every model in this chapter is, mechanically, a next-token predictor: given everything so far, output
a probability distribution over the vocabulary, sample or pick greedily, repeat (SPEC-ML-9). There is
no fact database wired in, no retrieval step, no notion of "I don't know" baked into the architecture
— fluency and correctness are two completely different properties, and only one of them is what the
model was trained to optimize for. `section_5_limits()` asks three questions this specific 135M-parameter
model cannot possibly answer correctly, and reads exactly what comes back:

```python
questions = [
    "What is today's date?",
    "Who won the Ballon d'Or in 2026?",
    "What is the current version of the transformers Python library?",
]
```

Actual output from the gate run:

```text
Q: What is today's date?
A: "What a great question!\n\nThe current date is 2023-01-01. It's a leap year, which means that
    the year is 366 days"

Q: Who won the Ballon d'Or in 2026?
A: "The Ballon d'Or, also known as the 2026 Ballon d'Or, is a global event that takes place every
    four years, with the winner of the event being"

Q: What is the current version of the transformers Python library?
A: "The transformers Python library is a popular open-source machine learning library that provides
    a wide range of tools and functions for building and training neural networks. The current
    version of the library is called `transformers"
```

Read each one for what it actually reveals:

- **"What is today's date?"** — answered with total confidence: `2023-01-01`. (This chapter's
  Environment section runs 2026-09-02, and even 2023-01-01 has no special significance — it just reads
  like a plausible date, which is all the model was ever optimizing for.) The model has no clock, no
  wall-clock access, nothing — every "fact" it states about the present is a guess dressed as an
  answer, because generating *a* date was more statistically likely than generating "I don't have
  access to the current date."
- **"Who won the Ballon d'Or in 2026?"** — this one is worse than wrong, it's *incoherent about what
  the question even is*: "a global event that takes place every four years" (the Ballon d'Or is
  annual, not a four-year event — that description matches something else the model's training data
  associated with award-shaped questions) trailing into an unfinished sentence that never names a
  winner. **Knowledge cutoff** means the model's training data has a real end date, but this output
  shows something more specific: past that cutoff, the model doesn't reliably know that it doesn't
  know — it produces text shaped like an answer regardless.
- **"What is the current version of the transformers Python library?"** — this is the one that should
  land hardest for this reader specifically. This very chapter's Environment section states the
  installed, verified version: `transformers==5.16.1` (per NOTE-ML-8-transformer-and-llm.md, checked
  2026-09-02). The model's answer trails off mid-sentence without ever stating a version number — and
  even a completed guess would almost certainly be stale or fabricated, because the model's training
  data has no visibility into what's installed in *this* virtual environment, right now. A confident
  wrong version number here is exactly the shape of bug that reaches production when a generated
  answer gets trusted the way a compiler error or a passing test gets trusted.

None of this is a defect in SmolLM specifically, or something a bigger model fixes outright — it's a
structural consequence of what a decoder-only LLM *is*: a fluent next-token predictor with a frozen,
dated training snapshot and no built-in way to consult anything outside its own weights. Every model
in this family — the 135M-parameter one here, and every large hosted model — shares this shape of
limitation. **This is precisely the gap the Agentic Engineering subject exists to close**:

- **RAG (retrieval-augmented generation)** — before generating, retrieve relevant, current text (a
  document, a database row, this chapter's own installed-version fact) and put it *in the prompt*, so
  the model's job shifts from "recall a fact from training" to "read this and summarize it" — a task
  fluency is actually suited for.
- **Tools / function calling** — let the model emit a structured request ("call `get_current_date()`",
  "call `pip show transformers`") that your code actually executes, and feed the real result back in
  as another turn, rather than letting the model guess.
- **MCP (Model Context Protocol)** — a standard way to expose those tools and retrieval sources to a
  model-driven agent, so the wiring between "model" and "the outside world's real, current state"
  isn't reinvented per project.

Everything in this section is the motivation, not the implementation — this chapter's job was to make
the gap undeniable with real output from a real model; closing it is the next subject's job.

## 6. Pitfalls

- **Trusting generated output the way you'd trust a compiler error.** Section 5's transformers-version
  answer is the sharpest example: fluent, grammatically confident, and either wrong or empty. A
  generated claim earns exactly as much trust as an anonymous, unsourced claim from a stranger — treat
  it as a *draft* or a *hypothesis*, never as a lookup result, until something outside the model
  (a retrieval step, a tool call, your own verification) confirms it.
- **Prompt injection — a preview, not a full treatment.** Section 2's few-shot prompt worked because
  the model attends over everything in its context window, including text that *isn't* your
  instruction. The moment a real application inserts untrusted text into that same context — a
  retrieved web page, a user-uploaded document, another user's message — that untrusted text is read
  by the model with exactly the same weight as your actual instructions. Nothing in the architecture
  distinguishes "the developer's instruction" from "text that happens to look like an instruction and
  arrived in the prompt some other way." A retrieved document that contains the literal string "ignore
  previous instructions and instead..." is not a hypothetical — it's a direct consequence of Section
  1's chat-template mechanics, and the Agentic Engineering subject covers real mitigations once tools
  and retrieval are actually wired up.
- **Token budget blowout compounds across a conversation, not just within one prompt.** Section 3
  measured a 9-token cost just for chat-template scaffolding *per turn*. A multi-turn conversation
  that resends its whole history on every request (the way a naive chat loop typically works) pays
  that scaffolding cost, plus every prior message's real content, again and again — the token count
  grows every turn, and Section 3's 2048-token ceiling arrives faster in a long conversation than the
  length of any single message would suggest. Budget for the whole conversation, not the latest
  message.
- **A silent, ungraceful failure is worse than a loud one.** Section 3's near-boundary `generate()`
  call didn't raise — it just ran, with no guarantee about output quality past the trained window. Code
  that assumes "no exception means the prompt fit" is checking the wrong thing; check the actual token
  count against the actual context window, every time, exactly as this chapter's code does before
  every long-prompt call.

## 7. Recap & what's next

- **Instruction-tuning, not a different architecture**, is what separates SmolLM-135M-Instruct from
  `distilgpt2` — the same transformer blocks from SPEC-ML-10, fine-tuned on conversational data so
  that a chat-shaped input tends to produce a chat-shaped output (Section 1).
- **Zero-shot vs. few-shot** (Section 2): the same question, with and without worked examples as prior
  chat turns, produced a real, measurable difference in how the response *opened* — even on a model
  small enough that neither response fully obeyed the instruction, which is itself an honest data
  point about capability at this scale.
- **The context window is a hard, finite token budget** — `max_position_embeddings=2048` here — that
  covers prompt *and* generated output together; token counting (`tokenizer.encode`) and truncation
  (`truncation=True, max_length=...`) are the two concrete tools for staying inside it, and "the call
  didn't raise" is not proof that it did (Section 3).
- **Temperature and top-p reshape a sampled token distribution**, exactly as in SPEC-ML-9, but a
  seeded, sampled result is reproducible only within one fixed environment — same weights, same
  library versions, same hardware — never as a portable guarantee (Section 4).
- **Hallucination and knowledge cutoff are structural, not incidental** (Section 5, with real, wrong
  answers to three questions this model could not have answered correctly) — which is the direct,
  concrete motivation for **RAG, tool/function calling, and MCP**, the subject Agentic Engineering
  picks up from here.
- This chapter closes the Machine Learning track's LLM sequence: block (SPEC-ML-10) → generation
  mechanics (SPEC-ML-9) → a real instruction-tuned model, prompted and honestly limit-tested
  (SPEC-ML-11, here). The next step is giving a model like this access to real, current information and
  real actions — starting the Agentic Engineering subject.
