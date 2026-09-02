# Representations: tokenizers, embeddings, similarity, quantization, fine-tuning

Every model you'll meet in this book — from a text classifier to an LLM-based agent — takes text
in and does linear algebra on it. Nothing about a transformer or a dense layer understands the
string `"queen"`. Before any of that math happens, text has to become numbers, and *how* it
becomes numbers determines almost everything downstream: how big your vocabulary is, whether
"king" and "queen" end up near each other in vector space, how much RAM a model needs, and whether
fine-tuning it is a laptop-sized job or a cluster-sized one.

This chapter covers the five ideas that recur everywhere else in the ML and Agentic subjects:

1. **Tokenizers** — splitting text into the units a model actually sees.
2. **Embeddings & word2vec** — turning tokens into dense vectors where geometry encodes meaning.
3. **Cosine similarity vs Euclidean distance** — how "closeness" between vectors gets measured,
   and why one of the two dominates in practice.
4. **Quantization** — shrinking a model's numbers (fp32 → fp16 → int8) to save memory and time.
5. **Fine-tuning** — adapting a pretrained model (full fine-tune vs LoRA), and when reaching for a
   prompt or a RAG pipeline instead is the better call.

If you've worked with Lucene, Elasticsearch, or written a hand-rolled lexer for a DSL, tokenizers
will feel familiar. If you've ever used a `HashMap<String, double[]>` to cache feature vectors,
embeddings will feel familiar too — the difference is that these vectors are *learned*, and
distance between them is meaningful.

All claims below trace to `research/NOTE-ML-4-representations.md` (checked 2026-09-02); the runnable
demo is [`code/similarity_demo.py`](code/similarity_demo.py). Run it yourself:

```bash
.venv-ml/Scripts/python.exe "Machine Learning/Theory/code/similarity_demo.py"
```

It uses real 384-dimensional sentence embeddings from `sentence-transformers` (version 6.0.1,
installed in this project's `.venv-ml`) if the model can load; if sentence-transformers or its
model download is unavailable, it falls back automatically to small, deterministic hand-built
vectors and says so on stdout. Every number quoted in this chapter is copied straight from an
actual run using the real model — nothing here is invented.

## 1. Text → numbers: tokenizers

### Concept

A **tokenizer** splits a string into the discrete units — tokens — that a model's vocabulary is
built from. There are three levels of granularity:

- **Word tokenizers** split on whitespace/punctuation (`"the cat sat"` → `["the", "cat", "sat"]`).
  Simple, but the vocabulary explodes: every inflection (`run`, `runs`, `running`, `runner`) is a
  separate entry, and any word the tokenizer never saw during training becomes an
  out-of-vocabulary (OOV) token — total information loss for that word.
- **Character tokenizers** split into individual characters. Tiny vocabulary (dozens of symbols),
  but sequences get very long, and a model has to work much harder to recover word-level meaning
  from character soup.
- **Subword tokenizers** — the modern default — split into pieces bigger than a character but
  smaller (or equal to) a word: `"unhappiness"` might become `["un", "happi", "ness"]`. This is
  the sweet spot: a bounded vocabulary size, reasonable sequence lengths, and no OOV problem,
  because any unseen word can still be spelled out of known pieces.

**Byte Pair Encoding (BPE)** is the standard algorithm for *learning* a subword vocabulary. It's
iterative and purely frequency-driven: start with every word split into characters, repeatedly find
the most frequent adjacent pair of symbols in the whole corpus, merge that pair into one new
symbol, and repeat until you hit a target vocabulary size or merge count
[source: BPE overview](https://towardsdatascience.com/byte-pair-encoding-for-beginners-708d4472c0c7/)
(checked 2026-09-02; NOTE-ML-4 evidence item 2, itself citing Sennrich et al., *Neural Machine
Translation of Rare Words with Subword Units*, 2016). GPT-2/GPT-3 use BPE variants; BERT uses a
close relative called WordPiece — different implementations, same idea (NOTE-ML-4 caveat: don't
assume two BPE-family tokenizers produce identical tokens for the same text).

If you've written a hand-rolled lexer, the loop shape is familiar — scan, find the best match,
consume, repeat — except a compiler's lexer applies **fixed** rules you wrote, while BPE **learns**
its merge rules from the frequency statistics of a training corpus. Nobody hand-writes the rule
"merge 'e' and 'r'"; the algorithm discovers it because `"er"` happens to be common.

### Worked example

`similarity_demo.py`'s `run_bpe_demo()` trains a tiny BPE tokenizer on a 5-word toy corpus and
prints every merge step. Each word starts as individual characters plus an end-of-word marker
(`</w>`, standard BPE bookkeeping so merges never cross a word boundary):

```text
toy corpus (word: frequency): {'low': 5, 'lowest': 2, 'newer': 6, 'wider': 3, 'new': 2}
merge 1: ('e', 'r') (count=9) -> 'er'
merge 2: ('er', '</w>') (count=9) -> 'er</w>'
merge 3: ('n', 'e') (count=8) -> 'ne'
merge 4: ('ne', 'w') (count=8) -> 'new'
merge 5: ('l', 'o') (count=7) -> 'lo'
merge 6: ('lo', 'w') (count=7) -> 'low'
merge 7: ('new', 'er</w>') (count=6) -> 'newer</w>'
merge 8: ('low', '</w>') (count=5) -> 'low</w>'
```

Watch what falls out for free: `"newer"` (frequency 6, the most common word) becomes a single
merged symbol, `"low"` becomes its own reusable subword shared between `"low"` and `"lowest"`, and
`"er</w>"` — a word-final "-er" — becomes a symbol shared between `"newer"` and `"wider"`.

The real payoff is applying those *same learned merges* to words the tokenizer never trained on:

```text
'slower'     -> ['s', 'low', 'er</w>']
'wildest'    -> ['w', 'i', 'l', 'd', 'e', 's', 't', '</w>']
```

`"slower"` was never in the training corpus, but it decomposes cleanly into `"s"` + the learned
`"low"` subword + the learned `"er</w>"` suffix. A pure word-level tokenizer would have had no
entry for `"slower"` at all and would have emitted a single `<UNK>` token, discarding the word
completely. That's the concrete answer to LO1's "why subword tokenizers won": bounded vocabulary,
*and* graceful handling of words nobody trained on.

Here's the merge-counting step in isolation, so you can see the mechanics without the loop:

```python
from collections import Counter

# word -> corpus frequency, each word split into characters + an end-of-word marker
vocab = {("l", "o", "w", "e", "r", "</w>"): 6, ("n", "e", "w", "e", "r", "</w>"): 4}

pair_counts = Counter()
for word, freq in vocab.items():
    for i in range(len(word) - 1):
        pair_counts[(word[i], word[i + 1])] += freq

most_frequent = max(pair_counts, key=pair_counts.get)
print(f"most frequent adjacent pair: {most_frequent} (count={pair_counts[most_frequent]})")
print(f"BPE would merge it into a single new symbol: {most_frequent[0] + most_frequent[1]!r}")
```

Running it prints `most frequent adjacent pair: ('w', 'e') (count=10)` — `"w"` followed by `"e"`
appears in both `lower` (×6) and `newer` (×4), so it wins the very first merge round.

## 2. Embeddings & word2vec: geometry encodes meaning

### Concept

An **embedding** is a dense, fixed-length vector (typically 100–1000+ floats) assigned to a token
or a piece of text, learned so that vectors for semantically similar inputs end up near each other.
Contrast this with a one-hot vector (`"cat"` = a vector of all zeros except a single 1) — one-hot
vectors carry no relationship information at all: every word is exactly as "far" from every other
word. Embeddings put "cat" and "dog" closer to each other than either is to "spreadsheet".

**word2vec** (Mikolov et al., Google, 2013) is the model that made this idea mainstream, via two
training setups (NOTE-ML-4 evidence item 3):

- **Skip-gram** — given a center word, predict its surrounding context words.
- **CBOW** (Continuous Bag-of-Words) — the reverse: given context words, predict the center word.
  Faster to train, slightly lower embedding quality.

Neither setup is told anything about grammar or meaning directly — the vectors that fall out of
training on "which words tend to appear near which other words" turn out to encode semantic and
even analogical relationships as *directions* in the vector space. The canonical illustration:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

"king minus man" isolates something like a "royalty" direction; adding "woman" lands you near
"queen" (NOTE-ML-4 evidence item 3). Modern embeddings (BERT-family, sentence-transformers) are
768-dimensional or more and come from transformer encoders rather than word2vec's shallow network,
but the same "geometry encodes meaning" property holds, which is what the worked example below
demonstrates with a real, current embedding model rather than the original word2vec.

If you know Java's `hashCode()`/`equals()`: an embedding is the opposite design goal.
`hashCode()` is explicitly allowed — encouraged — to scatter similar objects across the hash space.
An embedding is trained to do the reverse: pull semantically similar inputs *together*.

### Worked example

`similarity_demo.py` encodes a 14-word vocabulary with `sentence-transformers/all-MiniLM-L6-v2` — a
22M-parameter, ~44 MB, CPU-friendly model, Apache 2.0 licensed
[source: sentence-transformers on Hugging Face](https://huggingface.co/sentence-transformers)
(checked 2026-09-02; NOTE-ML-4 evidence item 6) — producing 384-dimensional unit vectors. It then
computes `king - man + woman`, re-normalizes it, and ranks every other word by cosine similarity:

```text
cos(analogy, queen     ) = 0.5795
cos(analogy, princess  ) = 0.4418
cos(analogy, prince    ) = 0.3882
cos(analogy, orange    ) = 0.3176
cos(analogy, france    ) = 0.3110
nearest neighbour: 'queen' (cos=0.5795) -- matches the expected geometry.
```

`"queen"` wins by a clear margin over the next-best candidate. The script asserts this — if a
future model swap ever broke the geometry, the run would fail loudly instead of silently printing
a wrong "expected" result.

The 2-D scatter below projects the same 14 embeddings down to 2 dimensions with PCA (implemented
as a plain SVD on centred data — no extra library, since PCA's components are exactly the right
singular vectors of the centred data matrix) purely so you can *see* the clustering:

![2-D PCA projection of word embeddings, coloured by category: royalty words cluster top-left, geography words cluster top-right, person/fruit/vehicle words cluster in separate groups below](artefacts/embedding_scatter_2d.png)

Royalty words (`king`, `queen`, `prince`, `princess`) cluster together; geography words (`paris`,
`france`, `london`, `england`) cluster together; fruit, vehicle, and person words each form their
own group — with no labels or rules telling the model any of that, just statistics from training
text. The title notes the top-2 principal components explain only 35.1% of total variance: this
2-D picture is illustrative, not the whole story — the real geometry lives in all 384 dimensions,
which is why the next section never does its math in this reduced space.

## 3. Similarity: cosine vs Euclidean

### Concept

Two standard ways to compare vectors (NOTE-ML-4 evidence item 1, citing
[Pinecone: vector similarity](https://www.pinecone.io/learn/vector-similarity/) and
[Zilliz: cosine vs Euclidean on normalized embeddings](https://zilliz.com/ai-faq/in-practical-terms-what-differences-might-you-observe-in-a-search-system-when-using-cosine-similarity-instead-of-euclidean-distance-on-the-same-set-of-normalized-embeddings),
both checked 2026-09-02):

**Cosine similarity** measures the angle between two vectors, ignoring their length:

```
cos_sim(a, b) = (a · b) / (‖a‖ ‖b‖)
```

Range `[-1, 1]`: `1` = pointing the same direction, `0` = orthogonal (unrelated), `-1` = opposite
directions. It's scale-invariant — multiply a vector by any positive constant and its cosine
similarity to everything else is unchanged.

**Euclidean distance** measures straight-line distance in the vector space:

```
d(a, b) = √(Σ(aᵢ - bᵢ)²)
```

Range `[0, ∞)`: `0` = identical vectors, larger = farther apart. Unlike cosine, this is sensitive
to magnitude, not just direction.

**The two are directly related for unit vectors.** If `‖a‖ = ‖b‖ = 1`:

```
d²(a, b) = 2 · (1 - cos_sim(a, b))
```

This is why embedding models (including `all-MiniLM-L6-v2` above) are so often trained or used to
output L2-normalized (unit-length) vectors: once vectors are unit length, ranking by cosine
similarity and ranking by Euclidean distance give the **same ordering** — the identity above is a
strictly decreasing function of cosine similarity. Cosine dominates in embedding search because
what a model learns is *direction* (semantic meaning); it usually doesn't intend vector length to
mean anything, so a distance metric that's sensitive to length adds noise cosine doesn't have.

### Worked example

`build_cosine_vs_euclidean_table()` computes both metrics for seven word pairs from the same
384-D unit vectors above, verifies `d² = 2·(1 - cos)` numerically for every pair (the script
`assert`s this — the run fails if it's ever off by more than `1e-6`), and renders the result:

![Table of cosine similarity, Euclidean distance, distance squared, and 2*(1-cosine) for seven word pairs, all matching, plus a highlighted pitfall row comparing king to king scaled 3x](artefacts/cosine_vs_euclidean_table.png)

Read the first row: `king`/`queen` have cosine similarity 0.6807 and Euclidean distance 0.7991.
Square that distance: `0.7991² = 0.6386`. Compute `2 × (1 - 0.6807) = 0.6386`. Same number, to four
decimal places, for every pair in the table — not a coincidence, the identity above guarantees it
for any pair of unit vectors.

The highlighted last row is the pitfall this identity warns about (and the subject of §6): it
compares `king` to `king` scaled by `3×` — same direction, three times the length. Cosine similarity
is `1.0000` (unchanged — cosine only sees direction). Euclidean distance is `2.0000` (it grew,
because the vectors are no longer unit length, so the `d² = 2(1-cos)` identity no longer applies —
notice the table correctly reports "n/a (not unit-norm)" in that row rather than a number that
would be wrong). If you naively used Euclidean distance to rank "how similar is X to king" without
normalizing first, a rescaled-but-identical-direction vector would look *maximally different*
instead of identical.

## 4. Quantization: fp32 → fp16 → int8

### Concept

Every number in a model — weights, and often the embeddings and activations too — is stored at
some numeric precision. Lowering that precision is **quantization**: fewer bits per number means
less memory and (with the right hardware) faster inference, at the cost of some numeric accuracy
(NOTE-ML-4 evidence item 4):

| Precision | Bytes/value | vs fp32 | Notes |
|---|---|---|---|
| **fp32** (single precision float) | 4 | baseline | Full precision, safest, largest. |
| **fp16** (half precision float) | 2 | 2× smaller | Needs hardware support (modern GPU tensor cores) for the speedup; minor accuracy loss. |
| **int8** (8-bit integer) | 1 | 4× smaller | Requires a calibration step (post-training quantization or quantization-aware training) to map floats into 256 integer levels without losing too much signal. |

[Source: quantization trade-offs](https://apxml.com/courses/cnns-for-computer-vision/chapter-8-model-compression-efficient-dl/quantization-reducing-precision)
(checked 2026-09-02; NOTE-ML-4 evidence item 4). int8 typically costs under 1% accuracy on many
models, but this is domain- and model-dependent — NOTE-ML-4's caveat is explicit that computer
vision models tend to tolerate int8 better than NLP models, whose activation distributions are
less uniform.

In Java terms, this is the same trade-off as choosing `double` vs `float` vs a fixed-point/byte
encoding for a large in-memory array: fewer bits per element means the array fits in cache/RAM more
easily and moves through memory faster, but every value now has less precision, and converting
*to* the smaller type is where you can lose information — the conversion back doesn't recover it.

### Worked example

`quantization_demo()` casts the same 14×384 embedding matrix used throughout this chapter through
all three precisions and measures, on this run's own vectors, both memory and how much cosine
similarity survives:

```text
--- quantization on the 14x384 embedding matrix ---
  fp32:  21.00 KiB  (baseline, 4 bytes/value)
  fp16:  10.50 KiB  (2 bytes/value, 2.0x smaller)  mean cosine vs fp32 = 1.000000
  int8:   5.25 KiB  (1 byte/value, 4.0x smaller)  mean cosine vs fp32 = 0.999938
```

fp16 loses essentially nothing here (rounding half-precision floats barely nudges a 384-D unit
vector's direction). The hand-rolled int8 quantizer — scale by the largest absolute value, round to
one of 256 levels, scale back (NOTE-ML-4 evidence item 4's post-training quantization, in its
simplest form) — still preserves 99.99% of the cosine similarity while using a quarter of the
memory. These specific numbers are measured on this chapter's own data, not a general benchmark
claim; the *sizes* (2× and 4× smaller) follow directly from the byte widths and generalize to any
array.

The same mechanics, isolated:

```python
import numpy as np

rng = np.random.default_rng(42)
weights_fp32 = rng.normal(0, 1, size=(1000,)).astype(np.float32)

weights_fp16 = weights_fp32.astype(np.float16)

scale = float(np.abs(weights_fp32).max())
weights_int8 = np.round(weights_fp32 / scale * 127).astype(np.int8)

print(f"fp32: {weights_fp32.nbytes} bytes")
print(f"fp16: {weights_fp16.nbytes} bytes ({weights_fp32.nbytes / weights_fp16.nbytes:.0f}x smaller)")
print(f"int8: {weights_int8.nbytes} bytes ({weights_fp32.nbytes / weights_int8.nbytes:.0f}x smaller)")
```

This prints `fp32: 4000 bytes`, `fp16: 2000 bytes (2x smaller)`, `int8: 1000 bytes (4x smaller)` —
the byte-width arithmetic is exact regardless of what the numbers represent.

## 5. Fine-tuning: full vs LoRA, and when to skip it

### Concept

**Fine-tuning** takes a pretrained model and continues training it on your own, usually much
smaller, dataset so it specializes to your task or domain. **Full fine-tuning** updates every
weight in the model — for a multi-billion-parameter LLM, that means storing gradients and optimizer
state for every one of those billions of parameters, which is where the GPU-memory cost comes from.

**LoRA** (Low-Rank Adaptation, Hu et al., 2021) is the standard parameter-efficient alternative: it
freezes all of the pretrained weights and, for each targeted linear layer's weight matrix `W`, adds
a trainable low-rank update:

```
ΔW ≈ A · Bᵀ        where A ∈ ℝ^(d_out × r), B ∈ ℝ^(d_in × r), r ≪ min(d_in, d_out)
```

Only `A` and `B` — the low-rank factors — are trained; `W` never moves. At inference time `A·Bᵀ` is
added back into `W` once, so there's no extra latency versus a fully fine-tuned model (Hu et al.,
2021; NOTE-ML-4 evidence item 5). NOTE-ML-4 reports that for a 7B-parameter LLM, LoRA adapters
typically add on the order of ~1M trainable parameters (~0.01% of the model) while remaining competitive with full
fine-tuning — but that whole-model figure only holds because LoRA is applied to a subset of layers
(usually the attention projections), with embeddings and most layers frozen entirely.

Zooming into a single linear layer makes the ratio concrete — not the whole-model 0.01%, just what
one layer costs:

```python
# A single linear layer in a 7B-parameter-class model: d_in = d_out = 4096 is a
# realistic hidden size for that scale (order-of-magnitude illustration only).
d_in, d_out, rank = 4096, 4096, 8

full_finetune_params = d_in * d_out
lora_params = rank * (d_in + d_out)  # trainable A (d_out x r) and B (d_in x r)

pct = lora_params / full_finetune_params * 100
print(f"full fine-tune trainable params for this one layer: {full_finetune_params:,}")
print(f"LoRA (rank={rank}) trainable params for this one layer: {lora_params:,}")
print(f"LoRA trains {pct:.3f}% as many parameters as full fine-tuning, for this layer")
```

This prints `full fine-tune trainable params for this one layer: 16,777,216`,
`LoRA (rank=8) trainable params for this one layer: 65,536`, and `LoRA trains 0.391% as many
parameters as full fine-tuning, for this layer`. That per-layer 0.391% and NOTE-ML-4's whole-model
~0.01% aren't in tension — the whole-model number is smaller because most of a real model's
parameters (embeddings, layers LoRA isn't applied to) are frozen and never enter either count.

If you've used Java's dependency injection to swap an implementation at runtime without touching
the interface: LoRA is closer to that than to recompiling the whole binary. `W` (the interface's
existing implementation) never changes; `A·Bᵀ` is a small, swappable adjustment layered on top —
and NOTE-ML-4's caveat is that this only fine-tunes what LoRA is attached to: layers left out of
the adapter (commonly the embeddings and early layers) stay exactly as pretrained, so full
fine-tuning still wins when you genuinely need to move every weight.

### Fine-tune vs prompt vs RAG

Fine-tuning is one of three ways to make a pretrained model behave the way you want, and it's
usually the most expensive:

- **Prompting** — just ask better, in the input. Zero training cost, changes take effect
  immediately, but the model's underlying knowledge doesn't change — it can only work with what's
  already in its weights plus whatever you put in the prompt.
- **RAG (Retrieval-Augmented Generation)** — look up relevant documents at query time and put them
  in the prompt. Solves "the model doesn't know about *my* data" without touching any weights —
  update the document store and the model's answers update immediately, no retraining. This is the
  Agentic Engineering subject's core worked example (a RAG app over PDFs) — this chapter is the
  theory that makes that later chapter's "we embed each document chunk, then retrieve by cosine
  similarity" step make sense: §2 and §3 above are exactly RAG's retrieval mechanism.
- **Fine-tuning** — actually change the weights. Worth it when the model needs a *behaviour* or
  *style* it can't reliably produce even with a good prompt and good retrieved context — a
  consistent output format, a specialized skill, a domain vocabulary saturating every response —
  not for "the model doesn't know this fact", which RAG solves more cheaply and more updatably.

A rough rule of thumb: reach for prompting first, RAG when the model needs facts it wasn't trained
on (and those facts change over time), and fine-tuning only when neither gets you the *behaviour*
you need, since fine-tuning is the only one of the three that requires training infrastructure,
takes real time to iterate, and produces a new artifact you have to version and redeploy.

## 6. Pitfalls

**Tokenizer/model mismatch.** A model was trained against one specific tokenizer's vocabulary and
merge rules. Encoding text with any other tokenizer — even one that looks similar, like BERT's
WordPiece vs GPT-2's BPE — produces a token sequence the model was never trained to interpret.
Always load the exact tokenizer that shipped with the model (NOTE-ML-4 caveat).

**Comparing unnormalized vectors with cosine "by feel", or Euclidean distance across differently
normalized embeddings.** §3's `king` vs `king×3` row is the sharp version of this: cosine ignores
magnitude by design, so it's safe on raw vectors — but Euclidean distance is not, and mixing
normalized and unnormalized vectors in the same Euclidean comparison silently corrupts the ranking.
NOTE-ML-4's caveat: always compare apples-to-apples, either both normalized or both raw, and know
which one your distance metric assumes.

**Over-quantizing.** §4's numbers looked almost free — 99.99% cosine similarity preserved at int8 —
but that was on 384-D sentence embeddings, which are comparatively forgiving. NOTE-ML-4's caveat is
explicit: accuracy impact is domain-dependent, and NLP models often tolerate int8 worse than vision
models do. Quantizing further than fp16/int8 (e.g. int4 and below) without calibration or
quantization-aware training is where accuracy loss stops being a rounding error and starts being a
real regression — always measure the accuracy delta on your own task, the way `quantization_demo()`
measured cosine-preserved on its own vectors rather than assuming a number.

**Fine-tuning when RAG (or a better prompt) would have solved it.** The single most common
overreach: reaching for a fine-tuning run to teach a model a fact, a document, or something that
changes over time. That's what RAG is for, at a fraction of the cost and with instant updates when
the underlying data changes — fine-tune for *behaviour*, not for *knowledge*.

## Recap & what's next

Text becomes tokens via a tokenizer (word/character/subword — BPE learns subwords from corpus
frequency, and handles unseen words by decomposing them into known pieces); tokens become dense
embeddings where distance is meaningful (word2vec's skip-gram/CBOW made this mainstream; the
"king − man + woman ≈ queen" geometry showed up again, unprompted, in a completely different,
modern embedding model); cosine similarity and Euclidean distance are two ways to measure that
distance, tied together by `d² = 2(1 − cos)` for unit vectors, with cosine dominating embedding
search because it's magnitude-invariant; quantization trades numeric precision for memory and
speed (fp32 → fp16 → int8); and fine-tuning (full or LoRA) is the most expensive of three ways to
change a model's behaviour, worth it only when prompting and retrieval genuinely can't get there.

Next up: **Image Classification (MNIST)** in Machine Learning → Worked Examples, where the first
real trained model in this book — a CNN, built on the neural-network fundamentals and architecture
choices from the earlier Theory chapters — gets applied to actual image data.
