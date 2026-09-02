# NOTE-ML-3: Network Architectures — CNN, RNN/LSTM/GRU, Transformer, Encoder-Decoder, Autoencoder

**Answer:** Convolution exploits weight-sharing and local connectivity for images (filters applied across spatial positions); LSTM/GRU use input/forget/output gates (LSTM) or update/reset gates (GRU) to control information flow and solve vanishing gradient in sequences; Transformer uses multi-head self-attention (learned query/key/value projections) + sinusoidal positional encodings (sin/cos at varying frequencies) to attend over sequences in parallel; encoder-decoder (seq2seq) encodes input sequence to context vector, decoder generates output conditioned on it; autoencoder compresses input through bottleneck layer to low-dimensional representation, then reconstructs—bottleneck forces useful feature extraction. scipy.signal.convolve2d(in1, in2, mode='full'|'same'|'valid', boundary='fill', fillvalue=0) convolves 2D arrays.

**Evidence:**

1. **Convolution: Weight Sharing & Local Connectivity** (verified 2026-09-02):
   - Local receptive field: filter (e.g., 5×5) connects to 5×5 region of input at each position
   - Weight sharing: same filter weights reused across all spatial positions; same parameters detect the same feature everywhere
   - Reduces parameter count vs dense layer; translation invariance (feature detected regardless of position)
   - Benefits: efficient training, spatial structure exploitation
   - Source: https://medium.com/@nerdjock/convolutional-neural-network-lesson-3-local-receptive-fields-and-weight-sharing-eb7af42343ff

2. **LSTM/GRU Gating Mechanisms** (verified 2026-09-02):
   - **LSTM gates:**
     - Input gate: controls what enters memory cell state
     - Forget gate: controls what leaves memory (multiplicative gating)
     - Output gate: controls what exits to next layer
     - Cell state maintains long-term memory; gating solves vanishing gradient by preserving gradients in additive connections
   - **GRU (simpler variant):**
     - Update gate: merged input/forget; how much previous memory to keep
     - Reset gate: combines previous memory with new input
     - 2 gates vs 3 (LSTM), more efficient, similar performance
   - **Vanishing gradient fix:** Gating mechanisms allow gradient to flow over long sequences without exponential decay
   - Sources: https://medium.com/@Hafiza_Shamza_Hanif/long-short-term-memory-lstm-and-gated-recurrent-unit-gru-overcoming-the-vanishing-gradient-dc67c07facb2 and https://ravjot03.medium.com/gru-explained-the-simplified-rnn-solution-for-sequential-data-c706d0d149c5

3. **Transformer: Self-Attention & Positional Encoding** (verified 2026-09-02):
   - **Self-attention mechanism (scaled dot-product attention):**
     - Query (Q), Key (K), Value (V) learned projections of input
     - Attention(Q, K, V) = softmax(Q×K^T / √d_k) × V
     - Allows each position to attend to all other positions; no sequential dependency like RNN
   - **Multi-head attention:** Compute attention multiple times in parallel with different learned projections; allows model to attend to different subspaces/representation positions simultaneously
   - **Positional encodings:**
     - Since no recurrence, position info must be injected: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
     - Sinusoids with geometric wavelengths (2π to 10000·2π); enables model to learn relative positions
   - **Parallelism:** Unlike RNNs (sequential), Transformer processes all positions in parallel per layer; enables fast training
   - Source: "Attention is all you need" (Vaswani et al., 2017) https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf and https://towardsdatascience.com/transformers-in-action-attention-is-all-you-need-ac10338a023a/

4. **Encoder-Decoder Architecture (Seq2Seq)** (verified 2026-09-02):
   - Encoder: RNN/LSTM/GRU/Transformer processes input sequence, produces context vector (final hidden state or weighted attention over all hidden states)
   - Decoder: RNN/LSTM/GRU/Transformer initialized with context vector, generates output sequence token-by-token (or in parallel for Transformer)
   - Attention mechanism bridges encoder→decoder: decoder attends to encoder hidden states to decide what input to focus on for each output token
   - Use cases: machine translation, summarization, speech recognition, any variable-length input→output task
   - Transformer version: both encoder and decoder use self-attention + cross-attention (decoder attends to encoder output)
   - Source: https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html and https://medium.com/@mervebdurna/exploring-seq2seq-encoder-decoder-and-attention-mechanisms-in-nlp-theory-and-practice-9b1022cf50b4

5. **Autoencoder: Bottleneck & Reconstruction** (verified 2026-09-02):
   - Architecture: Input → Encoder (shrink) → Bottleneck (low-dim latent) → Decoder (expand) → Output
   - Bottleneck layer has lower dimensionality than input; forces compression of information
   - Training objective: minimize reconstruction loss ||input - output||^2 (or cross-entropy for categorical)
   - Bottleneck dimensionality trade-off: smaller bottleneck → more compression but information loss; larger → more details retained but less dimensionality reduction
   - Representation learning: learned bottleneck representation captures salient features; useful for downstream tasks, denoising, anomaly detection
   - Source: https://www.v7labs.com/blog/autoencoders-guide and https://introml.mit.edu/notes/autoencoders.html

6. **scipy.signal.convolve2d API** (verified 2026-09-02):
   - Function signature: `scipy.signal.convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0)`
   - Parameters:
     - `in1, in2`: 2D arrays to convolve
     - `mode`: 'full' (default, output size = in1 + in2 - 1), 'same' (output same size as in1), 'valid' (output size = in1 - in2 + 1, no padding)
     - `boundary`: 'fill' (pad with fillvalue), 'symm' (symmetric), etc.
     - `fillvalue`: value to use for boundary padding (default 0)
   - Returns: 2D convolution result
   - Modern support: experimental Array API Standard compatibility (CuPy, PyTorch, JAX, Dask backends)
   - Example use: apply filter kernel (e.g., Sobel edge detector) to image
   - Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

**Caveats / limits:**

- **Convolution vs full connection:** Demo should emphasize parameter reduction (5×5×C filters vs (HW)×(HW) dense matrix for same input).
- **Pooling omitted from answer:** Max-pooling or avg-pooling is standard after conv layers but not core to "how conv works"; mention as complementary.
- **LSTM/GRU complexity:** Full gate equations (sigmoid/tanh compositions) are technical detail; block-level intuition sufficient.
- **Transformer training complexity:** Full training (mixed-precision, gradient accumulation) out of scope; focus on architecture.
- **Autoencoder unsupervised:** Unlike supervised nets, no label; reconstruction itself is the training signal.
- **Encoder-decoder seq-length mismatch:** If input and output sequences differ greatly in length, vanilla encoder-decoder (single context vector) bottleneck; attention mitigates.

**Recommendation:**

1. **CNN section:** Show a toy image (e.g., 8×8 with a single 3×3 filter); animate convolution sliding, highlight weight reuse.
2. **LSTM/GRU section:** Diagram the gates with boxes (input/forget/output) and arrows showing information flow; emphasize multiplicative gating ⊙ vs additive.
3. **Transformer section:** Illustrate multi-head attention as parallel heads with different Q/K/V projections; show PE formula visually (sinusoid wavelengths).
4. **Encoder-decoder section:** Draw input sequence → encoder hidden states → context/attention → decoder output sequence; label each component.
5. **Autoencoder section:** Sketch bottleneck as narrowing layer; emphasize that output reconstructs input (unlike classification where output is class label).
6. **Convolution demo code:** Use `scipy.signal.convolve2d(..., mode='same')` on small image (e.g., edge-detect kernel); show before/after.

## Correction (verified during authoring, 2026-09-03)
The autoencoder source `introml.mit.edu/notes/autoencoders.html` is DEAD (404) — verified during ML-2 authoring; use the still-live `v7labs.com/blog/autoencoders-guide` instead.
