"""conv_demo.py — SPEC-ML-2 worked example and figure generator.

Produces every artefact referenced by "Machine Learning/Theory/architectures.md":
  1. a hand-set 2D convolution (Sobel edge-detect kernels) applied to a
     synthetic sample image, via scipy.signal.convolve2d — grounded in
     NOTE-ML-3 evidence #1 and #6.
  2. a 2x2 max-pooling pass over the resulting feature map.
  3. schematic diagrams for: matching architecture to data shape, RNN vs
     LSTM vs GRU gating, transformer self-attention + positional encoding,
     and encoder-decoder / autoencoder — grounded in NOTE-ML-3 evidence
     #2-#5.

No training, no torch, no network access, no pip installs. The sample
image is synthesised in NumPy rather than loaded via ``scipy.datasets``,
because that module requires the optional ``pooch`` dependency, which is
not installed in the shared .venv-ml (confirmed by direct import check,
2026-09-02) and this project does not mutate that shared environment.

Environment (see NOTE-ML-1, and the installed versions printed below):
  numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.1

Run:
  .venv-ml/Scripts/python.exe "Machine Learning/Theory/code/conv_demo.py"
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")  # headless: write PNG files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import convolve2d

SEED = 42
ARTEFACT_DIR = pathlib.Path(__file__).resolve().parent.parent / "artefacts"
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Sample image + hand-set kernels
# ---------------------------------------------------------------------------

# Classic Sobel edge-detect kernels — one hand-set filter per direction.
# Each is a *fixed* 3x3 array of numbers we choose ourselves (this is what
# "hand-set kernel" means: no training, no learned weights).
SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)
SOBEL_Y = SOBEL_X.T  # same filter, rotated 90 degrees: vertical edges instead of horizontal


def make_sample_image(size: int = 40, seed: int = SEED) -> np.ndarray:
    """Synthesize a deterministic grayscale test image: a filled square plus
    a diagonal bar, with a touch of sensor noise. Clean geometric edges make
    it obvious, by eye, what a filter is responding to — a photo would not.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)
    img[8:24, 8:24] = 1.0  # filled square: gives a clean horizontal + vertical edge on each side
    for row in range(size):
        col = size - 1 - row
        lo, hi = max(col - 1, 0), min(col + 2, size)
        img[row, lo:hi] = 1.0  # a diagonal bar, anti-diagonal direction
    img += rng.normal(scale=0.03, size=img.shape)  # small seeded sensor noise
    return np.clip(img, 0.0, 1.0)


def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Hand-set 2D convolution via scipy.signal.convolve2d.

    mode="same" keeps the output the same H x W as the input (NOTE-ML-3 #6).
    boundary="symm" mirrors pixels at the border instead of the default
    zero-padding, so the image edge doesn't read as a fake, artificially
    strong feature.
    """
    return convolve2d(image, kernel, mode="same", boundary="symm")


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


def report_parameter_savings(image_size: int, kernel_size: int = 3) -> None:
    """Print the weight-sharing argument in concrete numbers (NOTE-ML-3 #1):
    a dense layer connecting every input pixel to every output pixel needs
    one weight per (input, output) pair; a conv filter reuses the same
    kernel_size x kernel_size weights at every spatial position.
    """
    n_pixels = image_size * image_size
    dense_weights = n_pixels * n_pixels
    conv_weights = kernel_size * kernel_size
    ratio = dense_weights / conv_weights
    print(f"[params] {image_size}x{image_size} image -> {image_size}x{image_size} feature map")
    print(f"[params] fully-connected (dense) layer: {n_pixels} x {n_pixels} = {dense_weights:,} weights")
    print(f"[params] {kernel_size}x{kernel_size} conv filter, weight-shared:  {conv_weights} weights")
    print(f"[params] dense uses {ratio:,.0f}x more weights than the shared conv filter")


# ---------------------------------------------------------------------------
# 2. Positional encoding (Vaswani et al., 2017 — NOTE-ML-3 #3)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. Small drawing helpers for the schematic diagrams
# ---------------------------------------------------------------------------

def _box(ax, xy, w, h, text, fc="#dbe9ff", ec="#2b4c7e", fontsize=8.5, style="round,pad=0.02,rounding_size=0.03"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.3,
                                 edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, zorder=3, wrap=True)


def _arrow(ax, start, end, color="#333333", lw=1.3, ls="solid", label=None, label_offset=(0, 0.03)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                  color=color, linewidth=lw, linestyle=ls, zorder=1))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="center", va="center",
                fontsize=7.5, color=color)


def _new_ax(figsize, title, xlim=(0, 10), ylim=(0, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Figure 1 — match architecture to data shape (the key decision)
# ---------------------------------------------------------------------------

def figure_data_shape_map(path: pathlib.Path) -> None:
    fig, ax = _new_ax((10, 6), "The key decision: what shape is your data?", xlim=(0, 12), ylim=(0, 10))

    _box(ax, (0.3, 4.2), 2.6, 1.6, "Your data", fc="#fff2cc", ec="#a67c00", fontsize=10)

    rows = [
        (7.6, "A grid: pixels on a 2D plane\n(image, spectrogram)", "CNN\n(convolution + pooling)", "#dbe9ff"),
        (5.6, "An ordered sequence,\nvariable length (text, time series)", "RNN / LSTM / GRU\nor Transformer", "#dbe9ff"),
        (3.6, "One sequence must become\na different sequence\n(translate, summarize)", "Encoder-Decoder\n(seq2seq)", "#dbe9ff"),
        (1.6, "You need a compressed\nrepresentation, not a label\n(anomaly detection, denoising)", "Autoencoder\n(bottleneck)", "#dbe9ff"),
    ]
    for y, shape_text, arch_text, fc in rows:
        _arrow(ax, (2.9, 5.0), (4.1, y + 0.75), color="#a67c00", lw=1.0)
        _box(ax, (4.1, y), 3.6, 1.5, shape_text, fc="#eef2f7", ec="#555555", fontsize=8)
        _arrow(ax, (7.7, y + 0.75), (8.6, y + 0.75))
        _box(ax, (8.6, y), 3.0, 1.5, arch_text, fc=fc, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Figure 2 — the convolution demo itself (original / Sobel-x / Sobel-y / magnitude)
# ---------------------------------------------------------------------------

def figure_convolution_demo(image: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                             magnitude: np.ndarray, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    panels = [
        (image, "Input image (40x40, synthetic)", "gray"),
        (gx, "After Sobel-X kernel\n(vertical edges)", "gray"),
        (gy, "After Sobel-Y kernel\n(horizontal edges)", "gray"),
        (magnitude, "Combined magnitude\nsqrt(Gx^2 + Gy^2)", "gray"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")
    fig.suptitle("Hand-set convolution: two 3x3 Sobel filters as edge detectors", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Figure 3 — pooling
# ---------------------------------------------------------------------------

def figure_pooling(feature_map: np.ndarray, pooled: np.ndarray, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(feature_map, cmap="gray")
    axes[0].set_title(f"Feature map, {feature_map.shape[0]}x{feature_map.shape[1]}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(pooled, cmap="gray")
    axes[1].set_title(f"After 2x2 max-pool, {pooled.shape[0]}x{pooled.shape[1]}", fontsize=10)
    axes[1].axis("off")
    fig.suptitle("Pooling: keep the strongest response per 2x2 block, halve the resolution",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Figure 4 — RNN vs LSTM vs GRU gating
# ---------------------------------------------------------------------------

def figure_rnn_lstm_gru(path: pathlib.Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    # --- Panel A: vanilla RNN — a repeated multiplicative chain ---
    ax = axes[0]
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Vanilla RNN — one path in, repeated multiplication out",
                 fontsize=11, fontweight="bold", loc="left")
    xs = [1.0, 4.0, 7.0, 10.0]
    for k, x in enumerate(xs):
        _box(ax, (x, 1.2), 1.6, 1.4, f"tanh cell\n$h_{{{k}}}$", fc="#f6d6d6", ec="#8a2b2b")
        if k > 0:
            _arrow(ax, (xs[k - 1] + 1.6, 1.9), (x, 1.9), color="#8a2b2b", label="x tanh'  <=1")
    ax.text(6, 0.2, "Backprop-through-time multiplies by tanh' (<=1) at every step -> the\n"
                     "gradient shrinks exponentially over long sequences (NOTE-ML-3 #2).",
            ha="center", fontsize=9, color="#8a2b2b")

    # --- Panel B: LSTM — additive cell-state highway + 3 gates ---
    ax = axes[1]
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.set_title("LSTM — a cell-state highway (additive) plus 3 gates", fontsize=11, fontweight="bold", loc="left")
    _arrow(ax, (0.6, 3.6), (11.4, 3.6), color="#2b6a3f", lw=2.2)
    ax.text(6, 3.9, "cell state $C_{t-1} \\rightarrow C_t$ : runs straight through, changed only by ADD/multiply gates",
            ha="center", fontsize=8.5, color="#2b6a3f")
    gate_x = [2.0, 5.3, 8.6]
    gate_names = ["Forget gate\n$\\sigma$ : keep or drop $C_{t-1}$",
                  "Input gate\n$\\sigma$ + tanh : write new info",
                  "Output gate\n$\\sigma$ : what leaves as $h_t$"]
    for x, name in zip(gate_x, gate_names):
        _box(ax, (x, 1.0), 2.4, 1.5, name, fc="#dcefe1", ec="#2b6a3f", fontsize=8)
        _arrow(ax, (x + 1.2, 2.5), (x + 1.2, 3.55), color="#2b6a3f")
    ax.text(6, 0.2, "Because the backbone is addition, not repeated multiplication, gradients\n"
                     "can flow across many steps without vanishing (NOTE-ML-3 #2).",
            ha="center", fontsize=9, color="#2b6a3f")

    # --- Panel C: GRU — fewer gates, single hidden state ---
    ax = axes[2]
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.0); ax.axis("off")
    ax.set_title("GRU — same idea, 2 gates instead of 3, no separate cell state",
                 fontsize=11, fontweight="bold", loc="left")
    _arrow(ax, (0.6, 3.2), (11.4, 3.2), color="#4a3b8a", lw=2.2)
    ax.text(6, 3.5, "hidden state $h_{t-1} \\rightarrow h_t$", ha="center", fontsize=8.5, color="#4a3b8a")
    gate_x = [3.2, 7.2]
    gate_names = ["Update gate\nhow much of $h_{t-1}$ to keep\n(merges RNN's input+forget)",
                  "Reset gate\nhow much of $h_{t-1}$\nto mix into the candidate"]
    for x, name in zip(gate_x, gate_names):
        _box(ax, (x, 0.7), 2.6, 1.6, name, fc="#e6e0f5", ec="#4a3b8a", fontsize=8)
        _arrow(ax, (x + 1.3, 2.3), (x + 1.3, 3.15), color="#4a3b8a")
    ax.text(6, 0.05, "2 gates vs LSTM's 3 -> fewer parameters, comparable performance in practice (NOTE-ML-3 #2).",
            ha="center", fontsize=9, color="#4a3b8a")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Figure 5 — transformer: parallel self-attention vs sequential RNN
# ---------------------------------------------------------------------------

def figure_transformer_attention(path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: RNN — strictly sequential
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("RNN: sequential\n(step t waits for step t-1)", fontsize=10.5, fontweight="bold")
    xs = [1.5, 4, 6.5, 9]
    for k, x in enumerate(xs):
        _box(ax, (x - 0.9, 3.2), 1.8, 1.2, f"$x_{{{k}}}$", fc="#f6d6d6", ec="#8a2b2b", fontsize=9)
        if k > 0:
            _arrow(ax, (xs[k - 1] + 0.9, 3.8), (x - 0.9, 3.8), color="#8a2b2b")
    ax.text(5.2, 1.6, "Layer t+1 cannot start until layer t finishes:\ntraining time grows with sequence length.",
            ha="center", fontsize=8.5, color="#8a2b2b")

    # Right: self-attention — every token attends to every token, in parallel
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Self-attention: parallel\n(every token attends to every token at once)",
                 fontsize=10.5, fontweight="bold")
    xs = [1.5, 4, 6.5, 9]
    ys = 6.4
    for k, x in enumerate(xs):
        _box(ax, (x - 0.9, ys), 1.8, 1.1, f"$x_{{{k}}}$", fc="#dbe9ff", ec="#2b4c7e", fontsize=9)
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            if i == j:
                continue
            _arrow(ax, (xi, ys - 0.05), (xj, ys - 0.05 - 0.001), color="#2b4c7e", lw=0.5)
    for k, x in enumerate(xs):
        ax.add_patch(FancyArrowPatch((x, ys - 0.1), (x, 4.6), arrowstyle="-", color="#2b4c7e", lw=0.6, alpha=0.5))
    ax.text(5.2, 3.9, "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V\n"
                       "computed for every pair at once; multi-head = several\n"
                       "such computations in parallel, different Q/K/V projections\n(NOTE-ML-3 #3).",
            ha="center", fontsize=8.5, color="#2b4c7e")
    ax.text(5.2, 1.1, "No recurrence -> no waiting -> the whole layer parallelizes on a GPU.",
            ha="center", fontsize=8.5, color="#2b4c7e")

    fig.suptitle("Why transformers displaced RNNs for sequence modelling", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_positional_encoding(pe: np.ndarray, path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(pe.T, cmap="RdBu", aspect="auto", origin="lower")
    ax.set_xlabel("position in the sequence")
    ax.set_ylabel("encoding dimension")
    ax.set_title("Sinusoidal positional encoding, PE(pos, i) — Vaswani et al. 2017 (NOTE-ML-3 #3)",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Figure 6 — encoder-decoder (seq2seq) and autoencoder
# ---------------------------------------------------------------------------

def figure_encoder_decoder_autoencoder(path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # --- Panel A: encoder-decoder / seq2seq ---
    ax = axes[0]
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Encoder-decoder (seq2seq): one sequence in, a different one out",
                 fontsize=11, fontweight="bold", loc="left")
    enc_xs = [0.8, 2.6, 4.4]
    for k, x in enumerate(enc_xs):
        _box(ax, (x, 2.4), 1.5, 1.1, f"enc$_{{{k}}}$", fc="#dbe9ff", ec="#2b4c7e", fontsize=8.5)
        if k > 0:
            _arrow(ax, (enc_xs[k - 1] + 1.5, 2.95), (x, 2.95), color="#2b4c7e")
    _box(ax, (6.1, 2.4), 1.7, 1.1, "context", fc="#fff2cc", ec="#a67c00", fontsize=8.5)
    _arrow(ax, (4.4 + 1.5, 2.95), (6.1, 2.95), color="#2b4c7e")
    dec_xs = [8.1, 9.7, 11.1]
    dec_w = [1.4, 1.3, 0.8]
    prev_right = 6.1 + 1.7
    for k, (x, w) in enumerate(zip(dec_xs, dec_w)):
        _box(ax, (x, 2.4), w, 1.1, f"dec$_{{{k}}}$", fc="#dcefe1", ec="#2b6a3f", fontsize=8.5)
        _arrow(ax, (prev_right, 2.95), (x, 2.95), color="#a67c00")
        prev_right = x + w
        _arrow(ax, (x + w / 2, 2.35), (x + w / 2, 1.5), color="#2b6a3f")
        ax.text(x + w / 2, 1.15, f"$y_{{{k}}}$", ha="center", fontsize=9)
    for x in enc_xs:
        _arrow(ax, (x + 0.75, 2.35), (x + 0.75, 3.7), color="#888888", ls="dashed", lw=0.7)
    ax.text(2.6, 4.05, "attention (dashed): decoder also looks back at\nevery encoder state, not only the final context",
            ha="center", fontsize=8, color="#666666")
    ax.text(2.6, 0.3, "Use case: machine translation, summarization — variable-length in, variable-length out.",
            ha="center", fontsize=8.5, color="#333333")

    # --- Panel B: autoencoder ---
    ax = axes[1]
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Autoencoder: compress, then reconstruct — no labels, just the input itself",
                 fontsize=11, fontweight="bold", loc="left")
    _box(ax, (0.4, 1.8), 1.6, 1.6, "input\n$x$", fc="#eef2f7", ec="#555555", fontsize=9)
    _box(ax, (2.6, 2.2), 1.5, 0.9, "encoder\n(shrinks)", fc="#dbe9ff", ec="#2b4c7e", fontsize=8)
    _box(ax, (4.7, 2.35), 1.1, 0.6, "bottleneck\n$z$", fc="#fff2cc", ec="#a67c00", fontsize=7.5)
    _box(ax, (6.4, 2.2), 1.5, 0.9, "decoder\n(expands)", fc="#dcefe1", ec="#2b6a3f", fontsize=8)
    _box(ax, (8.6, 1.8), 1.6, 1.6, "reconstruction\n$\\hat{x}$", fc="#eef2f7", ec="#555555", fontsize=9)
    _arrow(ax, (2.0, 2.6), (2.6, 2.65))
    _arrow(ax, (4.1, 2.65), (4.7, 2.65))
    _arrow(ax, (5.8, 2.65), (6.4, 2.65))
    _arrow(ax, (7.9, 2.65), (8.6, 2.6))
    ax.plot([1.2, 9.4], [1.55, 1.55], color="#8a2b2b", linewidth=1.1, linestyle="dashed")
    ax.add_patch(FancyArrowPatch((1.2, 1.55), (9.4, 1.55), arrowstyle="-|>", mutation_scale=10,
                                  color="#8a2b2b", linewidth=0.01))
    ax.text(5.2, 1.15, "loss compares input to reconstruction directly",
            ha="center", fontsize=7.5, color="#8a2b2b")
    ax.text(5.2, 0.5, "loss = ||x - x_hat||^2  (reconstruction error is the training signal, no label needed)",
            ha="center", fontsize=8.5, color="#8a2b2b")
    ax.text(5.2, 4.1, "bottleneck dim(z) << dim(x) forces the net to keep only what matters\n"
                       "-> denoising, anomaly detection, learned features for downstream tasks.",
            ha="center", fontsize=8.5, color="#333333")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)
    print(f"[env] numpy {np.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}")

    # --- convolution + pooling ---
    image = make_sample_image(size=40, seed=SEED)
    gx = apply_filter(image, SOBEL_X)
    gy = apply_filter(image, SOBEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    pooled = max_pool2x2(magnitude)

    figure_convolution_demo(image, gx, gy, magnitude, ARTEFACT_DIR / "02_conv_edge_detection.png")
    figure_pooling(magnitude, pooled, ARTEFACT_DIR / "03_pooling.png")
    report_parameter_savings(image_size=image.shape[0], kernel_size=3)

    # --- positional encoding (transformer) ---
    pe = positional_encoding(n_positions=50, d_model=32)
    figure_positional_encoding(pe, ARTEFACT_DIR / "06_positional_encoding.png")

    # --- schematics ---
    figure_data_shape_map(ARTEFACT_DIR / "01_data_shape_decision.png")
    figure_rnn_lstm_gru(ARTEFACT_DIR / "04_rnn_lstm_gru_gates.png")
    figure_transformer_attention(ARTEFACT_DIR / "05_transformer_attention.png")
    figure_encoder_decoder_autoencoder(ARTEFACT_DIR / "07_encoder_decoder_autoencoder.png")

    print(f"[artefacts] wrote 7 PNG files to {ARTEFACT_DIR}")
    for f in sorted(ARTEFACT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
