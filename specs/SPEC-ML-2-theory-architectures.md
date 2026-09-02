# SPEC-ML-2: Theory — Network Architectures (CNN, RNN, Transformer, Autoencoder)

**Status:** approved
**Subject:** Machine Learning
**Section:** Theory
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-1

## Intent
Explain the major layer types and architectures and, crucially, WHICH data shape each is built for —
so the reader can pick the right tool: CNNs for grids/images, RNN/LSTM/GRU for sequences, transformers
for attention over sequences, autoencoders for compression/representation, encoder–decoder for
seq-to-seq.

## Learning objectives
- LO1 — Explain convolution layers and why weight-sharing/locality suits images (vs a dense layer).
- LO2 — Explain LSTM/GRU: what recurrence and gating buy you for sequences, and the vanishing-gradient fix.
- LO3 — Explain the transformer at a block level (self-attention, positional encoding) and why it displaced RNNs.
- LO4 — Explain encoder–decoder and autoencoders (bottleneck, reconstruction) and their uses.

## Scope
In: conv, pooling; RNN→LSTM/GRU gating; transformer block (attention intuition); encoder–decoder; autoencoder. Diagrams over derivations.
Out: full attention math (→ ML-10 builds it), specific SOTA models.

## Outline
1. Match architecture to data shape — the key decision.
2. CNNs — convolution, filters, feature maps, pooling; a tiny conv-on-an-image demo.
3. Sequences — RNN limits → LSTM/GRU gates.
4. Transformers — self-attention intuition, positional encoding, parallelism; why they won.
5. Encoder–decoder & autoencoders — bottleneck, reconstruction, representation learning.
6. Pitfalls — using dense where conv fits, RNNs on long deps, over-deep nets.

## Assets to produce
- Prose: "Machine Learning/Theory/architectures.md"
- Code: "Machine Learning/Theory/code/conv_demo.py" (apply a hand-set convolution kernel to a sample image with numpy/scipy — CPU, no training)
- Artefacts: convolution-effect images (edge-detect kernel); architecture schematic diagrams (SVG/matplotlib).

## Claims to ground (Haiku, before writing)
- [ ] Verify the block-level descriptions of self-attention, LSTM/GRU gating, and autoencoders against authoritative sources (Attention Is All You Need; standard references).
- [ ] Confirm scipy/numpy 2D-convolution API used for the demo.

## Acceptance criteria
- [ ] AC1 — LOs delivered with the data-shape framing. AC2 — conv_demo.py runs and produces the convolution images; snippet-check passes. AC3 — architecture claims grounded. AC4 — each architecture tied to "what problem shape it solves".

## Gates
Entry: approved; notes landed. Exit: DoD checklist.
