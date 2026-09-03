"""Transformer internals from scratch — scaled dot-product attention, multi-head
attention, and one full transformer block, built directly on PyTorch primitives
(no `nn.MultiheadAttention`, no `nn.TransformerEncoderLayer`).

Formulas grounded in "Attention Is All You Need" (Vaswani et al., NeurIPS 2017),
https://arxiv.org/pdf/1706.03762 (Sections 3.1-3.3) — verified 2026-09-02, see
research/NOTE-ML-8-transformer-and-llm.md.

Environment (installed in .venv-ml, verified live 2026-09-02):
    torch==2.14.0+cpu
    numpy==2.5.2
    matplotlib==3.11.1
    Python 3.13 (.venv-ml)

Run:
    .venv-ml/Scripts/python.exe "Machine Learning/Worked Examples/llms/code/transformer_from_scratch.py"

Everything below runs on CPU, on tensors small enough to print and read by eye.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — write PNGs, never try to open a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Scaled dot-product attention — the formula from Section 3.2.1 of the paper
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Shapes (the last two dims are what matter; any leading dims — batch, heads —
    are carried through untouched, exactly like a Java generic type parameter):
        q: (..., n, d_k)   — n "query" positions, each a d_k-dim vector
        k: (..., m, d_k)   — m "key" positions, same feature width as q
        v: (..., m, d_v)   — m "value" positions, d_v can differ from d_k
        mask: broadcastable to (..., n, m), 1 = keep, 0 = block. Optional.
    Returns:
        output: (..., n, d_v)   — one d_v-dim vector per query position
        weights: (..., n, m)    — the attention matrix; each row sums to 1

    [source: Vaswani et al. 2017, "Attention Is All You Need", Section 3.2.1]
    (https://arxiv.org/pdf/1706.03762) (checked 2026-09-02)
    """
    d_k = q.size(-1)
    # QK^T: for every query, a raw similarity score against every key.
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Anywhere mask == 0, force the score to -inf so softmax sends it to ~0.
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # normalize each query's row to a probability distribution
    output = torch.matmul(weights, v)
    return output, weights


def demo_scaled_dot_product_attention() -> None:
    print("\n=== 1. Scaled dot-product attention (tiny, unbatched) ===")
    torch.manual_seed(SEED)
    n, d_k, d_v = 4, 3, 2  # 4 tokens, 3-dim keys/queries, 2-dim values
    q = torch.randn(n, d_k)
    k = torch.randn(n, d_k)
    v = torch.randn(n, d_v)
    print(f"Q shape: {tuple(q.shape)}  K shape: {tuple(k.shape)}  V shape: {tuple(v.shape)}")

    output, weights = scaled_dot_product_attention(q, k, v)
    print(f"attention weights shape: {tuple(weights.shape)}  (n_queries x n_keys)")
    print("attention weights (each row sums to 1):")
    print(weights.round(decimals=3))
    print("row sums:", weights.sum(dim=-1).round(decimals=3))
    print(f"output shape: {tuple(output.shape)}  (n_queries x d_v)")

    # Why divide by sqrt(d_k)? Compare the raw-score variance with and without it.
    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = raw_scores / math.sqrt(d_k)
    print(
        f"\nvariance of QK^T (unscaled): {raw_scores.var():.3f}  "
        f"vs scaled by 1/sqrt(d_k)={1/math.sqrt(d_k):.3f}: {scaled_scores.var():.3f}"
    )


# ---------------------------------------------------------------------------
# 2. Multi-head attention — split into h heads, attend in parallel, concat back
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    [source: Vaswani et al. 2017, Section 3.2.2] (https://arxiv.org/pdf/1706.03762)
    (checked 2026-09-02). Implemented with nn.Linear (the W^Q/W^K/W^V/W^O
    projections) and the scaled_dot_product_attention above, per
    research/NOTE-ML-8-transformer-and-llm.md.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly across heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # per-head width; d_k == d_v here for simplicity

        # One big Linear per projection covers all heads at once (d_model -> d_model);
        # splitting into heads happens by *reshaping* that output, not by using h
        # separate small Linear layers — cheaper and what real implementations do.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_k)  # split last dim into (n_heads, d_k)
        return x.transpose(1, 2)  # swap seq and n_heads so heads act like a batch dim

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_heads, seq, d_k) -> (batch, seq, d_model) — the inverse of split_heads."""
        batch, n_heads, seq, d_k = x.shape
        x = x.transpose(1, 2)  # (batch, seq, n_heads, d_k)
        return x.contiguous().view(batch, seq, n_heads * d_k)

    def forward(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.split_heads(self.w_q(x_q))  # (batch, n_heads, seq, d_k)
        k = self.split_heads(self.w_k(x_k))
        v = self.split_heads(self.w_v(x_v))

        # mask broadcasts across the n_heads dim: (batch, 1, seq_q, seq_k)
        head_mask = mask.unsqueeze(1) if mask is not None else None
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask=head_mask)
        # attn_out: (batch, n_heads, seq, d_k)  attn_weights: (batch, n_heads, seq_q, seq_k)

        combined = self.combine_heads(attn_out)  # (batch, seq, d_model)
        output = self.w_o(combined)  # final linear projection back to d_model
        return output, attn_weights


def demo_multi_head_attention() -> None:
    print("\n=== 2. Multi-head attention (split heads / concat heads) ===")
    torch.manual_seed(SEED)
    batch, seq, d_model, n_heads = 2, 5, 8, 2
    x = torch.randn(batch, seq, d_model)
    print(f"input x shape: {tuple(x.shape)}  (batch, seq_len, d_model)")

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    q = mha.split_heads(mha.w_q(x))
    print(
        f"after W^Q projection + split_heads: {tuple(q.shape)}  "
        f"(batch, n_heads={n_heads}, seq_len, d_k={d_model // n_heads})"
    )

    output, attn_weights = mha(x, x, x)  # self-attention: Q, K, V all come from x
    print(f"attention weights shape: {tuple(attn_weights.shape)}  (batch, n_heads, seq_q, seq_k)")
    print(f"combined (post concat) + W^O output shape: {tuple(output.shape)}  (batch, seq_len, d_model)")
    assert output.shape == x.shape, "multi-head attention must preserve (batch, seq, d_model)"
    print("output shape matches input shape — attention is a shape-preserving layer.")


# ---------------------------------------------------------------------------
# 3. Position-wise feed-forward network — FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """[source: Vaswani et al. 2017, Section 3.3] (https://arxiv.org/pdf/1706.03762)
    (checked 2026-09-02). Two Linear layers with a ReLU between them, applied
    identically (same weights) to every position independently.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


# ---------------------------------------------------------------------------
# 4. One transformer (encoder) block — attention + residual + LayerNorm + FFN
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Post-LN transformer encoder block, the arrangement in the original paper:
        x'  = LayerNorm(x  + MultiHeadAttention(x, x, x))
        x'' = LayerNorm(x' + FeedForward(x'))

    [source: Vaswani et al. 2017, Sections 3.1 and 3.3] (https://arxiv.org/pdf/1706.03762)
    (checked 2026-09-02; NOTE-ML-8 also documents the modern Pre-LN variant, not
    used here — this chapter implements the original paper's arrangement).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, tuple[int, ...]]]]:
        shapes: list[tuple[str, tuple[int, ...]]] = [("input x", tuple(x.shape))]

        attn_out, attn_weights = self.attn(x, x, x, mask=mask)
        shapes.append(("MultiHeadAttention(x, x, x)", tuple(attn_out.shape)))

        residual1 = x + attn_out  # residual / skip connection
        shapes.append(("x + attn_out (residual)", tuple(residual1.shape)))

        x = self.norm1(residual1)
        shapes.append(("LayerNorm(residual1)", tuple(x.shape)))

        ffn_out = self.ffn(x)
        shapes.append(("FeedForward(x)", tuple(ffn_out.shape)))

        residual2 = x + ffn_out
        shapes.append(("x + ffn_out (residual)", tuple(residual2.shape)))

        x = self.norm2(residual2)
        shapes.append(("LayerNorm(residual2) = block output", tuple(x.shape)))

        return x, attn_weights, shapes


def demo_transformer_block() -> list[tuple[str, tuple[int, ...]]]:
    print("\n=== 3. One transformer block — shapes at every step ===")
    torch.manual_seed(SEED)
    batch, seq, d_model, n_heads, d_ff = 2, 5, 8, 2, 16
    x = torch.randn(batch, seq, d_model)

    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    output, attn_weights, shapes = block(x)

    for step_name, shape in shapes:
        print(f"  {step_name:<38s} -> {shape}")
    assert output.shape == x.shape, "a transformer block never changes (batch, seq, d_model)"
    print("block output shape == block input shape (attention + FFN are shape-preserving).")

    # Save the shapes table as a CSV artefact — the "shapes-through-the-block table" asset.
    csv_path = ARTEFACTS_DIR / "shapes_through_block.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "output_shape (batch, seq_len, d_model)"])
        for step_name, shape in shapes:
            writer.writerow([step_name, str(shape)])
    print(f"\nwrote shapes table -> {csv_path}")
    return shapes


# ---------------------------------------------------------------------------
# 5. Positional encoding + causal mask, visualised as an attention heatmap
# ---------------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    [source: Vaswani et al. 2017, Section 3.5] (https://arxiv.org/pdf/1706.03762)
    (checked 2026-09-02). Returns a fixed (not learned) (seq_len, d_model) tensor,
    added elementwise to token embeddings so the model can tell *position* apart —
    otherwise attention (a weighted sum over all positions) is fully permutation-
    invariant, like a Java `Set<Token>` that has forgotten the original order.
    """
    position = torch.arange(seq_len).unsqueeze(1).float()  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # even dims
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
    return pe


def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular mask: position i may attend to positions 0..i, never i+1..end.

    Shape (seq_len, seq_len); 1 = allowed, 0 = blocked. Passed to
    scaled_dot_product_attention, which turns the 0s into -inf before softmax.
    [source: Vaswani et al. 2017, Section 3.2.3 "Masked Multi-Head Attention"]
    (https://arxiv.org/pdf/1706.03762) (checked 2026-09-02).
    """
    return torch.tril(torch.ones(seq_len, seq_len))


def demo_positional_encoding_and_causal_mask() -> None:
    print("\n=== 4. Positional encoding + causal mask -> attention heatmap ===")
    torch.manual_seed(SEED)

    tokens = ["The", "cat", "sat", "on", "mat"]
    vocab = {w: i for i, w in enumerate(tokens)}
    seq_len, d_model, n_heads = len(tokens), 8, 2
    token_ids = torch.tensor([[vocab[w] for w in tokens]])  # (1, seq_len) batch of 1

    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=d_model)
    tok_emb = embedding(token_ids)  # (1, seq_len, d_model)
    print(f"token ids shape: {tuple(token_ids.shape)}  token embeddings shape: {tuple(tok_emb.shape)}")

    pe = positional_encoding(seq_len, d_model)  # (seq_len, d_model)
    print(f"positional encoding shape: {tuple(pe.shape)}  (broadcasts over the batch dim)")
    x = tok_emb + pe  # (1, seq_len, d_model) — embeddings now carry position

    mask = causal_mask(seq_len)  # (seq_len, seq_len)
    print("causal mask (1 = attend, 0 = blocked):")
    print(mask.int())

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    _, attn_weights = mha(x, x, x, mask=mask.unsqueeze(0))  # unsqueeze -> (1, seq_len, seq_len), broadcasts over heads
    print(f"masked attention weights shape: {tuple(attn_weights.shape)}  (batch, n_heads, seq_q, seq_k)")

    head0 = attn_weights[0, 0].detach().numpy()  # first batch element, first head: (seq_len, seq_len)
    print("head 0 attention weights (rows are queries, cols are keys; upper triangle ~0 from the mask):")
    print(attn_weights[0, 0].round(decimals=3))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(head0, cmap="viridis", vmin=0, vmax=head0.max())
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("key position (attending TO)")
    ax.set_ylabel("query position (attending FROM)")
    ax.set_title("Causal self-attention, head 0\n(upper triangle blocked by the mask)")
    for i in range(seq_len):
        for j in range(seq_len):
            ax.text(j, i, f"{head0[i, j]:.2f}", ha="center", va="center",
                     color="white" if head0[i, j] < head0.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    png_path = ARTEFACTS_DIR / "attention_heatmap.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nwrote attention heatmap -> {png_path}")


def main() -> None:
    torch.manual_seed(SEED)
    demo_scaled_dot_product_attention()
    demo_multi_head_attention()
    demo_transformer_block()
    demo_positional_encoding_and_causal_mask()
    print("\nAll demos completed. Artefacts written to:", ARTEFACTS_DIR)


if __name__ == "__main__":
    main()
