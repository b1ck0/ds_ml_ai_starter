# Network Architectures — Matching the Tool to the Data's Shape

*Machine Learning · Theory · SPEC-ML-2*

## What & why

In Java, picking a data structure is half the design: a `HashMap` for lookups, a `TreeMap` for
ordered traversal, an `ArrayDeque` for a queue. Pick wrong and the code still compiles — it's just
slow, or awkward, or wrong in ways that only show up under load. Neural-network architectures are
the same kind of decision, one level up: **the "data structure" you pick has to match the shape of
the data you're feeding it.**

[ML-1: Neural Network Fundamentals](01-neural-network-fundamentals.md) built the base unit — a neuron,
a dense (fully connected) layer, gradient descent, backprop. A dense layer treats its input as one
flat, unordered vector of numbers. That's a fine default, but it throws away structure that's often
the whole point:

- A **grid** (an image's pixels, a spectrogram) has 2D locality — a pixel's neighbours matter more
  than a pixel on the other side of the image. A dense layer doesn't know that; it happily connects
  every pixel to every other pixel from scratch.
- A **sequence** (a sentence, a time series) has order — swapping two words changes the meaning. A
  dense layer has no notion of "before" and "after."
- Some problems need a **compressed representation** of the input, not a label. Some need to turn
  **one sequence into a different sequence** (translate a sentence, summarize a document).

This chapter is the map: four families of architecture, one paragraph each on *what problem shape
they solve*, then enough of the mechanism to reason about them — diagrams over derivations, per this
chapter's scope. The one runnable worked example is a hand-set 2D convolution, because seeing a
filter slide over an image and light up its edges says more than three paragraphs about "spatial
feature extraction."

## 1. The key decision — match architecture to data shape

Before opening a deep-learning framework, ask: **what shape is my data, and what shape is my
output?** That single question narrows the field to one or two candidates almost every time.

![Decision map: data shape to architecture](artefacts/01_data_shape_decision.png)

*Figure 1 — the decision this chapter teaches you to make. Reproduced by
[`conv_demo.py`](code/conv_demo.py), function `figure_data_shape_map`.*

| Data shape | Architecture | Section |
|---|---|---|
| A grid — pixels on a 2D plane | **CNN** (convolution + pooling) | §2 |
| An ordered, variable-length sequence | **RNN / LSTM / GRU**, or a **Transformer** | §3, §4 |
| One sequence must become a different sequence | **Encoder-decoder** (seq2seq) | §5 |
| You need a compressed representation, not a label | **Autoencoder** (bottleneck) | §5 |

Keep this table in view for the rest of the chapter — every section below is one row of it, expanded.

## 2. CNNs — convolution, filters, feature maps, pooling

**What problem shape it solves:** your input is a grid with local structure (an image, most
commonly), and the same visual feature — an edge, a corner, a texture — can appear anywhere in that
grid. You want a layer that (a) looks at small local neighbourhoods, not the whole image at once, and
(b) reuses the same feature detector at every position instead of learning a separate one for each
pixel.

### Concept

A **convolution layer** slides a small matrix of numbers — a **kernel** or **filter** (say, 3x3) —
across the input, and at each position computes one number: the sum of the kernel's weights
multiplied by the pixels underneath it. Two things fall out of that description, and they're the
whole reason CNNs exist for images:

- **Local connectivity** — each output value depends only on a small local patch of the input (the
  kernel's **receptive field**), not the entire image, the way a dense layer's output would.
- **Weight sharing** — the *same* 3x3 = 9 numbers are reused at every position in the image. If a
  filter learns to detect a vertical edge, it detects vertical edges everywhere, not just in the
  top-left corner. This is also what gives CNNs (approximate) **translation invariance**: shift the
  input a few pixels, and the same features still get detected.
  [source: local receptive fields & weight sharing](https://medium.com/@nerdjock/convolutional-neural-network-lesson-3-local-receptive-fields-and-weight-sharing-eb7af42343ff)
  (checked 2026-09-02); NOTE-ML-3 evidence #1.

Weight sharing is also a massive parameter-count win over a dense layer connecting the same input to
an output of the same size — see the worked example below for the actual numbers on our sample
image.

Applying one kernel produces one **feature map** — a 2D grid of "how strongly did this filter fire
at this position." A real conv layer learns many kernels in parallel (edge detectors, corner
detectors, and combinations no human would hand-design), producing a stack of feature maps, one per
filter.

**Pooling** (most commonly **max-pooling**) is a fixed, unlearned downsampling step usually applied
after a conv layer: slide a small window (2x2 is typical) over a feature map and keep only the
strongest value in each window, halving the height and width. It has no weights of its own — it's a
deliberate resolution reduction, not a filter (NOTE-ML-3 caveats). It buys two things: fewer numbers
to carry into the next layer, and a bit more translation tolerance (a feature that moved one pixel
still survives the pool).

### Worked example — a hand-set edge-detect kernel

The full script is [`code/conv_demo.py`](code/conv_demo.py). It builds a small 40x40 synthetic
grayscale image (a filled square plus a diagonal bar, with a touch of seeded noise), hand-sets two
classic **Sobel** edge-detect kernels — one for vertical edges, one for horizontal — and convolves
each over the image with `scipy.signal.convolve2d` (NOTE-ML-3 evidence #6;
[source: scipy.signal.convolve2d docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
(checked 2026-09-02), confirmed against the scipy 1.18.1 installed in this project's `.venv-ml`).
Note that these kernel weights are *hand-set*, not learned — exactly the point: a real CNN starts
from random weights and *learns* kernels like these via backprop, but the mechanics of "slide a
small matrix, multiply-and-sum" are identical either way.

```python
import numpy as np
from scipy.signal import convolve2d

# Classic Sobel edge-detect kernels — a fixed 3x3 array of numbers we choose
# ourselves. This is what "hand-set kernel" means: no training involved.
SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)
SOBEL_Y = SOBEL_X.T  # the same filter, rotated 90 degrees: horizontal edges instead of vertical


def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Hand-set 2D convolution via scipy.signal.convolve2d.

    mode="same" keeps the output the same H x W as the input (NOTE-ML-3 #6).
    boundary="symm" mirrors pixels at the border instead of the default
    zero-padding, so the image edge doesn't read as a fake, artificially
    strong feature.
    """
    return convolve2d(image, kernel, mode="same", boundary="symm")
```

Running the script produces the before/after images:

![Sobel edge detection on a synthetic image](artefacts/02_conv_edge_detection.png)

*Figure 2 — the same 40x40 image after each hand-set kernel. The vertical-edge kernel (Sobel-X)
lights up the square's left/right sides and misses its top/bottom; the horizontal-edge kernel
(Sobel-Y) does the opposite. `sqrt(Gx^2 + Gy^2)` combines both into one "edge strength" map — this is
exactly what stacking multiple learned filters into multiple feature maps buys a real CNN: different
filters detecting different things, combined downstream.*

Then a 2x2 max-pool halves the resolution of the combined feature map:

```python
def max_pool2x2(feature_map: np.ndarray) -> np.ndarray:
    """A stride-2, 2x2 max-pool: keep the strongest response in each 2x2
    block, halving height and width. Unlike a conv filter, pooling has no
    learned (or even hand-set) weights — it's a fixed downsampling rule.
    """
    h, w = feature_map.shape
    h2, w2 = h - h % 2, w - w % 2
    trimmed = feature_map[:h2, :w2]
    blocks = trimmed.reshape(h2 // 2, 2, w2 // 2, 2)
    return blocks.max(axis=(1, 3))
```

![Max-pooling a feature map](artefacts/03_pooling.png)

*Figure 3 — the edge map before and after 2x2 max-pooling: half the resolution, same shapes still
clearly visible, because pooling keeps the strongest ("most confident") response per block.*

**The parameter-count argument, in real numbers.** The script also prints a direct comparison: for
this 40x40 image, a dense layer connecting every input pixel to every output pixel would need
`1600 x 1600 = 2,560,000` weights. The 3x3 Sobel kernel needs **9** weights, reused at all 1,600
positions — a **~284,000x** reduction, for this image size:

```console
[params] 40x40 image -> 40x40 feature map
[params] fully-connected (dense) layer: 1600 x 1600 = 2,560,000 weights
[params] 3x3 conv filter, weight-shared:  9 weights
[params] dense uses 284,444x more weights than the shared conv filter
```

That ratio only grows with image size — a 224x224 image (a typical CNN input) would need over
2.5 *billion* dense weights for the same one-output-per-pixel mapping. Weight sharing isn't a minor
optimisation; for image-shaped data it's the difference between trainable and not.

To reproduce all of the above yourself:

```console
.venv-ml/Scripts/python.exe "Machine Learning/Theory/code/conv_demo.py"
```

## 3. Sequences — RNN limits, then LSTM/GRU gating

**What problem shape it solves:** your input is an ordered sequence of variable length — words in a
sentence, ticks in a time series — where what came *before* affects how you should interpret what
comes *now*. A dense layer (fixed-size input, no notion of order) doesn't fit; you need a layer that
carries information forward from one step to the next.

### Concept — the vanilla RNN, and why it struggles

A **recurrent neural network (RNN)** processes a sequence one element at a time, maintaining a
**hidden state** `h_t` that gets updated at every step: roughly, `h_t = tanh(W_x x_t + W_h h_{t-1})`
— a small dense layer applied at each step, fed both the current input and the *previous* hidden
state. That hidden state is the network's memory of everything it has seen so far in the sequence.

The problem shows up when you try to train it. Backpropagation-through-time has to send the error
signal backward through every one of those steps, and at each step it multiplies by the derivative
of `tanh` — a number that's always `<= 1`, often much smaller. Multiply a small number by itself
across, say, fifty timesteps, and the gradient reaching the earliest steps is effectively zero: the
network can't learn dependencies that span more than a handful of steps. This is the **vanishing
gradient problem** for sequences (NOTE-ML-3 evidence #2;
[source: LSTM & GRU overcoming vanishing gradient](https://medium.com/@Hafiza_Shamza_Hanif/long-short-term-memory-lstm-and-gated-recurrent-unit-gru-overcoming-the-vanishing-gradient-dc67c07facb2)
(checked 2026-09-02)) — the same multiplicative-shrinkage mechanism [ML-1] showed for deep stacks of
`sigmoid`/`tanh` layers (NOTE-ML-2 evidence #4), just unrolled across *time* instead of *depth*.

### Concept — LSTM and GRU: gating fixes it

**LSTM** (Long Short-Term Memory) fixes this by giving the network a second, separate channel: the
**cell state**, `C_t`, which runs across timesteps mostly *unchanged* except for explicit add/erase
operations controlled by three learned **gates** (each a small sigmoid layer, output in `[0, 1]`,
acting as a "how much" dial):

- **Forget gate** — how much of the old cell state `C_{t-1}` to keep vs. discard.
- **Input gate** — how much new information to write into the cell state.
- **Output gate** — how much of the (possibly updated) cell state to expose as this step's hidden
  state `h_t`.

The reason this fixes vanishing gradients: the cell-state backbone is **additive**
(`C_t = forget ⊙ C_{t-1} + input ⊙ candidate`), not the repeated-multiplication chain a vanilla RNN's
hidden state goes through. Gradients can flow back across many steps along that additive path without
shrinking exponentially (NOTE-ML-3 evidence #2).

**GRU** (Gated Recurrent Unit) is a simpler variant: it merges the forget/input gates into a single
**update gate** and adds a **reset gate**, and drops the separate cell state entirely — two gates
instead of three, fewer parameters, and comparable performance in practice
([source: GRU explained](https://ravjot03.medium.com/gru-explained-the-simplified-rnn-solution-for-sequential-data-c706d0d149c5)
(checked 2026-09-02); NOTE-ML-3 evidence #2).

Full gate equations are outside this chapter's scope — the takeaway to carry forward is the
block-level shape: *an additive memory path, with learned gates deciding how much to keep, write, and
expose.*

![RNN vs LSTM vs GRU gating](artefacts/04_rnn_lstm_gru_gates.png)

*Figure 4 — top: a vanilla RNN's hidden state is repeatedly multiplied step to step, so gradients
shrink. Middle: LSTM's cell state runs straight through (green line), touched only by additive/gated
operations from three gates. Bottom: GRU, the same idea with two gates and no separate cell state.
Reproduced by [`conv_demo.py`](code/conv_demo.py), function `figure_rnn_lstm_gru`.*

## 4. Transformers — self-attention, positional encoding, parallelism

**What problem shape it solves:** you still have a sequence, but you want two things RNNs can't
give you cheaply: (1) *any* position can directly influence *any* other position, no matter how far
apart, without the signal having to survive N sequential hops; and (2) the whole layer can be
computed in parallel, not step-by-step, because sequential processing is what makes RNNs slow to
train on long sequences and on GPUs (which are built for parallel work, not for waiting on step `t-1`
before starting step `t`).

### Concept — self-attention

Instead of recurrence, a **transformer** layer lets every position in the sequence look directly at
every other position and decide how much attention to pay to each, in one parallel computation. For
every token, the model learns three projections of it: a **Query** (what this token is looking for),
a **Key** (what this token offers, for other tokens to match against), and a **Value** (what this
token actually contributes if attended to). Attention is then:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

— compare every query against every key (`Q K^T`), scale, turn into a probability distribution over
positions (`softmax`), and use those probabilities to take a weighted blend of all the values. No
step waits on another step: this whole computation is one matrix multiplication.
**Multi-head attention** runs several of these in parallel with independently-learned Q/K/V
projections, so different heads can pick up on different kinds of relationships (e.g. one head
tracking subject-verb agreement, another tracking nearby words) at once.
(NOTE-ML-3 evidence #3; [source: "Attention Is All You Need," Vaswani et al.,
2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
(checked 2026-09-02), and
[a walkthrough of the mechanism](https://towardsdatascience.com/transformers-in-action-attention-is-all-you-need-ac10338a023a/)
(checked 2026-09-02).)

### Concept — positional encoding

There's a catch: attention treats the sequence as an unordered set of positions — nothing in the
formula above knows that token 3 comes before token 5. Since there's no recurrence to implicitly
encode order, the transformer has to be told position explicitly, so it adds a **positional
encoding** vector to each token's embedding before the first attention layer. The original scheme
uses sine and cosine waves at geometrically varying frequencies, one pair of frequencies per pair of
embedding dimensions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

(NOTE-ML-3 evidence #3, same source as above.) The formula is computed once, deterministically — no
training involved — and simply added to the input embeddings. Here it is computed for real (50
positions, 32 encoding dimensions) and plotted:

```python
import numpy as np


def positional_encoding(n_positions: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Source: "Attention Is All You Need" (Vaswani et al., 2017) — NOTE-ML-3 #3.
    """
    pos = np.arange(n_positions)[:, None].astype(np.float64)
    i = np.arange(d_model)[None, :].astype(np.float64)
    angle_rates = 1.0 / np.power(10000.0, (2 * np.floor(i / 2)) / d_model)
    angles = pos * angle_rates
    pe = np.zeros((n_positions, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe
```

![Sinusoidal positional encoding heatmap](artefacts/06_positional_encoding.png)

*Figure 5 — the actual `PE` matrix for 50 positions x 32 dimensions. Low dimensions (bottom rows)
oscillate fast — they distinguish nearby positions; high dimensions (top rows) oscillate slowly —
they distinguish coarse, far-apart position ranges. Every position gets a distinct combined pattern,
which is exactly what lets a model with no recurrence still infer relative position.
Reproduced by [`conv_demo.py`](code/conv_demo.py), function `figure_positional_encoding`.*

### Concept — why transformers displaced RNNs

Put self-attention and positional encoding together and you get the two wins RNNs couldn't offer: no
sequential bottleneck (every position's attention is computed in one parallel pass, so training scales
far better with hardware and with sequence length), and a direct path between any two positions
(an RNN has to relay information through every intermediate step; a transformer connects position 1
to position 500 in a single attention computation) (NOTE-ML-3 evidence #3).

![RNN sequential processing vs transformer parallel self-attention](artefacts/05_transformer_attention.png)

*Figure 6 — left: an RNN's hidden state chain — step t+1 cannot start before step t finishes. Right:
self-attention — every token attends to every other token in one parallel computation; multi-head
attention runs several such computations side by side.
Reproduced by [`conv_demo.py`](code/conv_demo.py), function `figure_transformer_attention`.*

Full attention math, masking for autoregressive decoding, and building a transformer block from
scratch are out of scope here — that's ML-10.

## 5. Encoder-decoder and autoencoders — bottleneck and reconstruction

Two more shapes share a family resemblance — both have an "encoder" stage and a "decoder" stage —
but solve very different problems, so it's worth being precise about which is which.

### Encoder-decoder (seq2seq) — one sequence becomes a different sequence

**What problem shape it solves:** the input and output are both sequences, but they're not the same
sequence — translation (English in, French out), summarization (long document in, short summary
out), speech recognition (audio in, text out). Lengths on each side can differ arbitrarily.

**Concept:** an **encoder** (RNN/LSTM/GRU, or a transformer) reads the whole input sequence and
produces a **context** — historically a single vector (the encoder's final hidden state), summarizing
"everything about the input." A **decoder** is then initialized from that context and generates the
output sequence one token at a time, each new token conditioned on what it has generated so far plus
the context. A single fixed-size context vector is a bottleneck when the input is long — the fix is
**attention**: instead of compressing everything into one vector, the decoder attends back to *every*
encoder position at each output step, deciding at generation time which parts of the input matter
most for the token it's producing right now (NOTE-ML-3 evidence #4;
[source: seq2seq and attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)
(checked 2026-09-02), and
[a worked walkthrough](https://medium.com/@mervebdurna/exploring-seq2seq-encoder-decoder-and-attention-mechanisms-in-nlp-theory-and-practice-9b1022cf50b4)
(checked 2026-09-02)). A transformer-based encoder-decoder uses this same shape, but both encoder
self-attention and encoder-decoder cross-attention are the parallel kind from §4, not a single
context vector.

### Autoencoder — compression and reconstruction, no labels needed

**What problem shape it solves:** you don't have a label at all — you want the network to learn a
*compressed representation* of the input, useful for dimensionality reduction, denoising, anomaly
detection, or as a feature extractor for a downstream model.

**Concept:** an **autoencoder** is an encoder that shrinks the input down to a low-dimensional
**bottleneck** (the latent representation `z`), followed by a **decoder** that expands `z` back out,
trying to reconstruct the original input as closely as possible. There's no external label — the
training signal is the reconstruction error itself, typically `loss = ||x - x_hat||^2`: how far the
reconstruction `x_hat` is from the original `x` (NOTE-ML-3 evidence #5;
[source: the autoencoders guide](https://www.v7labs.com/blog/autoencoders-guide) (checked
2026-09-02)). The bottleneck being *smaller* than the input is the whole mechanism: the network is
forced to discard whatever it can, keeping only the features that let it reconstruct the input well —
those surviving features are, by construction, the salient ones. A smaller bottleneck means more
compression (and more information loss); a larger one retains more detail but reduces the
dimensionality-reduction benefit — a trade-off you set, not one the network resolves for you.

![Encoder-decoder (seq2seq) and autoencoder schematics](artefacts/07_encoder_decoder_autoencoder.png)

*Figure 7 — top: encoder-decoder — an input sequence is encoded, summarized into a context, and a
decoder generates a differently-shaped output sequence; dashed arrows show attention reaching back to
every encoder state. Bottom: autoencoder — the same "encode then decode" shape, but the target is the
*input itself*, and the point is the narrow bottleneck in the middle, not the output sequence length.
Reproduced by [`conv_demo.py`](code/conv_demo.py), function `figure_encoder_decoder_autoencoder`.*

**The distinction worth keeping straight:** encoder-decoder's decoder produces a *new, different*
sequence (a translation); an autoencoder's decoder tries to reproduce the *same* input it started
from. Same two-stage shape, opposite goals.

## 6. Pitfalls

- **Using a dense layer where a conv layer fits.** The worked example above isn't decorative — for a
  224x224 image, a dense layer mapping pixels to pixels would need billions of weights where a 3x3
  conv filter needs 9. If your input is grid-shaped and you reach for `Dense`/`Linear` layers
  end-to-end, you're both throwing away the locality structure in the data *and* paying for it in
  parameters you don't need (NOTE-ML-3 evidence #1). **How to see it:** print your model's parameter
  count for a dense-only version vs. a conv version at the same input size, the way `conv_demo.py`
  does — the ratio is usually startling.

- **Expecting an RNN (even LSTM/GRU) to handle arbitrarily long-range dependencies.** Gating fixes
  *vanishing* gradients, it doesn't make sequence length free — very long sequences still stress an
  RNN's single hidden-state bottleneck, and training remains inherently sequential (NOTE-ML-3
  evidence #2, #3). **How to see it:** if performance degrades sharply as your sequences get longer,
  or training is unacceptably slow because you can't parallelize across timesteps, that's the signal
  to reach for a transformer or, at minimum, add attention over the encoder states rather than relying
  on a single final hidden state.

- **Stacking layers deep without a fix for vanishing gradients.** The exact mechanism from §3
  (repeated multiplication by a small derivative shrinking the gradient) isn't unique to *time* — it
  happens across *depth* too: composing `L` sigmoid/tanh layers multiplies by a term `<= 0.25` at each
  one, so `(0.25)^L -> 0` as `L` grows (NOTE-ML-2 evidence #4, from [ML-1]). This is exactly why ReLU
  activations and gated architectures (LSTM/GRU, and — one layer further — the residual connections
  transformers use) matter more as networks get deeper. **How to see it:** if a deep network's early
  layers' weights barely move during training while later layers train fine, suspect vanishing
  gradients, not a data problem.

## Recap & what's next

Four architecture families, four data shapes:

- **CNN** — grid data, weight-sharing exploits local, position-independent structure (§2).
- **RNN -> LSTM/GRU** — ordered sequences; gating fixes the vanilla RNN's vanishing-gradient limit by
  giving gradients an additive path back through time (§3).
- **Transformer** — sequences, without the sequential bottleneck: self-attention connects any two
  positions directly and computes in parallel; positional encoding restores the order information
  attention alone doesn't have (§4).
- **Encoder-decoder** and **autoencoder** — both an "encode then decode" shape, aimed at opposite
  goals: producing a *different* output sequence vs. compressing and reconstructing the *same* input
  (§5).

The common thread: none of this is arbitrary — every mechanism here (weight sharing, gating,
attention, bottlenecks) exists to make a specific data shape learnable efficiently. The reflex to
build going forward: look at your data's shape *first*, then pick the architecture family, not the
other way around.

Next: [ML-3 — Theory: Representations](03-representations.md) picks up where "what shape is the raw
input" leaves off, and asks how a network turns raw pixels, tokens, or categories into the numeric
vectors every architecture in this chapter actually operates on.
