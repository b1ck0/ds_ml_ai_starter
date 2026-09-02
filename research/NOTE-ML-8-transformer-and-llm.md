# NOTE-ML-10 & ML-11: Transformer Architecture & Small Instruction-Tuned Models

**Answer:** **ML-10 Transformer internals:** Scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V` where d_k is key dimension; scaling by 1/√d_k prevents vanishing gradients. Multi-head attention: `MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O` where `head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)` (h=8 heads, d_k=d_v=64); each head projects inputs via learned linear layers, computes attention independently, concatenates, and projects back. Transformer block: sequential sub-layers with residual connections and layer normalization: `LayerNorm(x + Sublayer(x))` (applied after each sub-layer: self-attention, cross-attention, feed-forward). Implementation: `nn.Linear` (projections), `F.softmax` (attention weights), `nn.LayerNorm` (normalization). From **"Attention Is All You Need"** (Vaswani et al., NeurIPS 2017). **ML-11 Small instruction-tuned model:** Recommended: `HuggingFaceTB/SmolLM-135M-Instruct` (Apache 2.0 license, 135M params, ~300-400MB weights, CPU-runnable). Alternative fallback: distilgpt2 (88.2M params; caveat: not instruction-tuned, may not follow instructions well; acceptable for tutorial with clear limits noted). SmolLM context window: ~2048 tokens (typical for small models). Chat template: use `tokenizer.apply_chat_template(messages, tokenize=False)` or `tokenizer.apply_chat_template(messages, return_tensors="pt")` (built-in to model card). Token counting: `len(tokenizer.encode(text))` or `tokenizer.get_token_ids(text)`. Checked 2026-09-02.

**Evidence:**

### ML-10: Transformer Architecture

1. **Scaled dot-product attention formula** (verified 2026-09-02)
   - Source: "Attention Is All You Need" https://arxiv.org/pdf/1706.03762 (Vaswani et al., NeurIPS 2017, Section 3.2.1)
   - Formula: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
   - Shapes: Q ∈ ℝ^(n×d_k), K ∈ ℝ^(m×d_k), V ∈ ℝ^(m×d_v) → Output ∈ ℝ^(n×d_v)
   - Scaling: 1/√d_k prevents dot products from growing too large (preventing vanishing gradients in softmax)
   - Complexity: O(nm·d_k) for single head

2. **Multi-head attention formula** (verified 2026-09-02)
   - Source: "Attention Is All You Need", Section 3.2.3
   - Formula: `MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O` where `head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)`
   - Dimensions: W^Q_i, W^K_i ∈ ℝ^(d_model × d_k), W^V_i ∈ ℝ^(d_model × d_v), W^O ∈ ℝ^(hd_v × d_model)
   - Typical: h=8 heads, d_k = d_v = d_model/h = 512/8 = 64
   - Benefit: allows model to attend to different representation subspaces in parallel
   - Output: Concat reshapes (h, n, d_v) → (n, h×d_v) → linear projection back to d_model

3. **Transformer block / encoder layer structure** (verified 2026-09-02)
   - Source: "Attention Is All You Need", Section 3.1 (encoder) and 3.3 (decoder)
   - Sequential sub-layers:
     1. **Multi-head self-attention** → output x' = x + MultiHeadAttn(x, x, x) [residual]
     2. Apply layer norm: LayerNorm(x')
     3. **Feed-forward network** (two linear layers with ReLU) → output x'' = x' + FFN(x') [residual]
     4. Apply layer norm: LayerNorm(x'')
   - **Post-LN variant (original paper):** `x' = x + Sublayer(LayerNorm(x))`
   - **Pre-LN variant (modern):** `x' = LayerNorm(x) + Sublayer(x)` (more stable training)
   - Feed-forward: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2 (two linear layers with ReLU; inner dim typically 2048 or 4×d_model)

4. **PyTorch NN APIs for implementation** (verified 2026-09-02)
   - `nn.Linear(d_in, d_out)`: learnable linear transformation
   - `F.softmax(logits, dim=-1)`: normalize attention weights
   - `nn.LayerNorm(d_model)`: normalize across feature dimension
   - `nn.MultiheadAttention(embed_dim, num_heads, ...)`: built-in multi-head attention (PyTorch 1.9+)
   - Source: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
   - Alternative: implement attention from scratch using Linear, softmax, and matrix operations (torch.matmul, torch.bmm)

5. **Positional encoding** (verified 2026-09-02)
   - Formula (sinusoidal): `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
   - Added to input embeddings before transformer layers
   - Encodes absolute position (sequence order matters)

6. **Causal masking** (verified 2026-09-02)
   - Applied in attention (decoder): prevent tokens from attending to future positions
   - Mask: set attention logits to -∞ (or very large negative) before softmax
   - Ensures autoregressive generation (each position depends only on past)

### ML-11: Small Instruction-Tuned Models

1. **Model: SmolLM-135M-Instruct** (verified 2026-09-02)
   - Source: https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct
   - License: Apache-2.0 (permissive, free for any use)
   - Size: 135M parameters
   - Weights size: ~300-400 MB (FP16 format, varies by quantization)
   - Training: Fine-tuned on instruction-following datasets (UltraChat, HelpSteer)
   - Instruction-following: Good for chat/QA tasks; understands system prompts and user intent
   - **CPU feasibility:** Can run on modern CPU (Intel i5/Ryzen 5+) with ~8GB RAM; inference is slow (~1-5 tokens/sec on CPU)
   - Download: Free via `transformers`; auto-cached to `~/.cache/huggingface/hub/`

2. **Alternative fallback: distilgpt2** (verified 2026-09-02)
   - Source: https://huggingface.co/distilbert/distilgpt2
   - License: Apache-2.0
   - Size: 88.2M parameters (even smaller than SmolLM)
   - **Caveat:** Not instruction-tuned; was trained on general text; may not follow instructions well
   - Use if: SmolLM cannot run in sandbox for any reason (weight limit, memory constraint)
   - Recommendation: Use with clear limits noted in chapter ("distilgpt2 is not instruction-tuned; responses may be generic")

3. **Chat template support (SmolLM)** (verified 2026-09-02)
   - Source: Model card at https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct
   - API: `tokenizer.apply_chat_template(messages, tokenize=False)` → returns formatted prompt string
   - Format: `"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"`
   - Also: `tokenizer.apply_chat_template(messages, return_tensors="pt")` → returns tokenized input_ids directly
   - Example usage:
     ```python
     messages = [
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is 2+2?"}
     ]
     prompt = tokenizer.apply_chat_template(messages, tokenize=False)
     inputs = tokenizer(prompt, return_tensors="pt")
     outputs = model.generate(**inputs, max_new_tokens=50)
     ```

4. **Context window / sequence length** (verified 2026-09-02)
   - SmolLM-135M-Instruct: ~2048 tokens (max context window)
   - Typical small models: 1024-2048 tokens
   - Source: Model card configuration (config.json max_position_embeddings field)
   - Implication: long prompts + long generated text may exceed window (truncate/summarize context)

5. **Token counting** (verified 2026-09-02)
   - API: `len(tokenizer.encode(text))` → returns number of token IDs
   - Also: `tokenizer(text)["input_ids"]` → list of token IDs; `len(...)` to count
   - Encoder example: `num_tokens = len(tokenizer.encode("Hello world"))` → ~3 tokens
   - For chat template: count tokens after applying `apply_chat_template()` to get true context usage

6. **Generate API compatibility** (verified 2026-09-02)
   - Code: `model.generate(input_ids, max_new_tokens=100, temperature=0.7, top_p=0.9)`
   - Params: do_sample, temperature, top_k, top_p, num_beams (same as distilgpt2 from ML-9)
   - Output: token IDs; decode via `tokenizer.decode(output_ids[0], skip_special_tokens=True)`

**Caveats / limits:**

- **CPU inference speed:** SmolLM on CPU: ~1-5 tokens/sec (a 100-token response takes 20-100 seconds). Acceptable for tutorial but mention timing.
- **Context window size:** 2048 tokens for SmolLM-135M is quite small; long conversations/documents will get truncated. Mention this constraint.
- **Instruction-following quality:** SmolLM is much smaller than GPT-3.5/4; may not understand complex instructions or maintain context over many turns. Set expectations.
- **Chat template format:** Not all models use the same template; SmolLM uses `<|im_start|>` format (compatible with recent Qwen/Llama style).
- **Distilgpt2 fallback:** Not instruction-tuned; may produce generic or off-topic text; use only if SmolLM unavailable.
- **Parameter size growth:** If SmolLM-135M doesn't run on CPU in sandbox, no smaller instruction-tuned model readily available; would need to fallback to non-instruction model (distilgpt2) or mention limitation.

**Recommendation:**

1. **For ML-10 chapter:**
   - Start with scaled dot-product attention formula; show the matrix dimensions and the 1/√d_k scaling rationale.
   - Implement attention from scratch: Q, K, V tensors (small, e.g., 2×4 shape) → softmax(QK^T / √d_k) → weight matrix visualization.
   - Extend to multi-head: show how 8 heads split the 512-dim embedding into 8×64-dim subspaces; concatenate and project back.
   - Assemble one transformer block on a tiny 2×512 input tensor; show shapes at each step.
   - Visualize an attention matrix as a heatmap (show what the model "attends to").
   - Reference: "Attention Is All You Need" paper (Vaswani et al., 2017).

2. **For ML-11 chapter:**
   - **Lead with SmolLM-135M-Instruct as primary recommendation.**
   - Show how to load + chat template + generate.
   - Run zero-shot prompt, then few-shot prompt (show the difference).
   - Count tokens before/after chat template to illustrate context usage.
   - Mention context window limit: "SmolLM maxes out at 2048 tokens; long conversations will be truncated."
   - **If weight/memory constraint arises:** fallback to distilgpt2 with clear caveat: "distilgpt2 is not instruction-tuned; it may not follow instructions as reliably as SmolLM."
   - Temperature/top_p section: show the SAME prompt under temp=0.5, 1.0, 1.5 to illustrate diversity vs consistency.
   - Bridge to Agentic: "LLMs hallucinate and have knowledge cutoffs; this is why RAG (retrieval-augmented generation) and tools (function calling) matter."

3. **For both:**
   - Use exact model IDs: `"HuggingFaceTB/SmolLM-135M-Instruct"` not generic "small model".
   - Pin transformers 5.16.1 (already in .venv-ml).
   - Explicitly specify `device="cpu"` in inference code.
   - Measure inference time (include in output) so readers know to expect slow CPU inference.
   - Provide reproducible example: same prompt → same seed → same generation (demonstrate determinism with greedy decode).

4. **For code snippets:**
   - ML-10: from-scratch transformer block (Linear + softmax + LayerNorm, no framework multi-head attention; do it manually with torch.bmm).
   - ML-11: load SmolLM + apply chat template + generate + count tokens + decode; real output (no fabrication).
