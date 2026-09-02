"""Representations -- companion code for
Machine Learning/Theory/representations.md (SPEC-ML-3).

Text has to become numbers before any model can touch it. This script is the runnable
half of that story, in three parts:

  1. A tiny byte-pair-encoding (BPE) tokenizer, implemented from scratch on a five-word
     toy corpus, showing subword merges emerge from frequency counts alone -- and then
     using the learned merges to tokenize a word the "tokenizer" never saw, which is
     the whole reason subword tokenizers beat whole-word vocabularies.
     Algorithm: research/NOTE-ML-4-representations.md, evidence item 2.

  2. Real sentence embeddings (or, failing that, small hand-built stand-ins -- see
     `get_embeddings()`) for a vocabulary of 14 words, used to:
       a. Reproduce the "king - man + woman ~= queen" analogy geometry by vector
          arithmetic + nearest-neighbour search (evidence item 3).
       b. Compute cosine similarity AND Euclidean distance on the same word pairs,
          verify the normalization identity d^2 = 2*(1 - cos) for unit vectors, and
          demonstrate the "magnitude fools Euclidean, cosine ignores it" pitfall by
          comparing a vector to a rescaled copy of itself (evidence item 1).
          Saves cosine_vs_euclidean_table.png.
       c. Project the 384-D (or 9-D, if the fallback is used) embeddings down to 2
          dimensions with PCA (implemented as a plain SVD on centred data -- no new
          library needed) and plot them, to make "geometry encodes meaning" visible.
          Saves embedding_scatter_2d.png.

  3. A quantization illustration: cast the same embedding matrix from fp32 down to
     fp16 and to a hand-rolled int8 (min-max scale, per NOTE-ML-4 evidence item 4),
     and measure the actual memory saved and cosine similarity preserved on OUR OWN
     vectors -- not a claimed benchmark number, an empirical one from this run.

Grounded facts (all from research/NOTE-ML-4-representations.md, checked 2026-09-02):
  - Cosine similarity, Euclidean distance formulas, and the unit-vector identity
    d^2(a,b) = 2*(1 - cos_sim(a,b)) -- evidence item 1, citing
    https://zilliz.com/ai-faq/in-practical-terms-what-differences-might-you-observe-in-a-search-system-when-using-cosine-similarity-instead-of-euclidean-distance-on-the-same-set-of-normalized-embeddings
    and https://www.pinecone.io/learn/vector-similarity/ (checked 2026-09-02).
  - BPE algorithm (iteratively merge the most frequent adjacent symbol pair into a new
    vocabulary symbol, repeat) -- evidence item 2, citing "Neural Machine Translation
    of Rare Words with Subword Units" (Sennrich et al., 2016) via
    https://medium.com/@varunsivamani/byte-pair-encoding-bpe-5fdced1b31cd and
    https://towardsdatascience.com/byte-pair-encoding-for-beginners-708d4472c0c7/.
    The five-word toy corpus below is this script's own invention, not from the paper.
  - Word2vec analogy geometry ("king - man + woman ~= queen") -- evidence item 3,
    citing https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp-comparison-between-cbow-and-skip-gram-models/.
    The demo below reproduces the geometry with real sentence embeddings (or the
    fallback hand-vectors), not the original word2vec model.
  - Quantization levels fp32/fp16/int8 and what each trades -- evidence item 4, citing
    https://apxml.com/courses/cnns-for-computer-vision/chapter-8-model-compression-efficient-dl/quantization-reducing-precision.
  - sentence-transformers (Apache 2.0) and all-MiniLM-L6-v2 as a CPU-friendly
    embedding model -- evidence item 6, citing https://huggingface.co/sentence-transformers
    and https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html.

Environment (all versions re-verified live against this project's .venv-ml on
2026-09-02, the same way NOTE-2-package-versions.md's numpy/matplotlib pins were):
    numpy==2.5.2, matplotlib==3.11.1, Python 3.13.7 (venv reports this; project
    targets 3.11+), sentence-transformers==6.0.1 (installed in .venv-ml; NOTE-ML-4
    evidence item 6 grounds the *choice* of sentence-transformers/all-MiniLM-L6-v2 as
    CPU-friendly, not this exact version number -- the version itself comes straight
    from `sentence_transformers.__version__` in this project's own environment, the
    live source for "what's actually installed here"). torch==2.14.0+cpu (NOTE-ML-1)
    is sentence-transformers' backend; this script never imports torch directly.

    sentence-transformers is an optional path: if it or its model download is
    unavailable, `get_embeddings()` catches the failure and falls back to small,
    hand-built 9-D vectors constructed so the same analogy geometry holds by
    construction (see `hand_made_embeddings()`). Either way the script prints which
    path it took and both artefacts are produced.

Run:
    python similarity_demo.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
RNG_SEED = 42

WORDS = [
    "king", "queen", "man", "woman", "prince", "princess",
    "apple", "orange", "car", "truck", "paris", "france", "london", "england",
]

# Word -> category, used only to colour the 2-D scatter plot.
CATEGORY = {
    "king": "royalty", "queen": "royalty", "prince": "royalty", "princess": "royalty",
    "man": "person", "woman": "person",
    "apple": "fruit", "orange": "fruit",
    "car": "vehicle", "truck": "vehicle",
    "paris": "geography", "london": "geography", "france": "geography", "england": "geography",
}
CATEGORY_COLOR = {
    "royalty": "tab:purple", "person": "tab:blue", "fruit": "tab:orange",
    "vehicle": "tab:gray", "geography": "tab:green",
}


# ---------------------------------------------------------------------------
# 1. BPE tokenizer, from scratch, on a toy corpus
# ---------------------------------------------------------------------------

def get_pair_counts(vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    """Count every adjacent symbol-pair frequency across the vocabulary, weighted by
    each word's corpus frequency. NOTE-ML-4 evidence item 2: "iteratively count all
    adjacent symbol pairs"."""
    pairs: dict[tuple[str, str], int] = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def merge_vocab(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    """Replace every occurrence of `pair` with a single merged symbol, everywhere in
    the vocabulary. NOTE-ML-4 evidence item 2: "replace most-frequent pair with new
    symbol, add to vocabulary"."""
    merged_symbol = pair[0] + pair[1]
    new_vocab: dict[tuple[str, ...], int] = {}
    for word, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def apply_bpe(word: str, learned_merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize a (possibly unseen) word by replaying the learned merges IN THE ORDER
    they were learned. This is what makes BPE handle out-of-vocabulary words: a word
    the algorithm never saw during training still decomposes into the subword pieces
    it did see."""
    symbols = list(word) + ["</w>"]
    for pair in learned_merges:
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


def run_bpe_demo(n_merges: int = 8) -> list[tuple[str, str]]:
    """Trains a tiny BPE tokenizer on a 5-word toy corpus (word -> frequency), printing
    every merge step, then applies the learned merges to words the tokenizer never
    trained on -- LO1's "why subword tokenizers won" made concrete: an out-of-vocabulary
    word still decomposes into familiar pieces instead of becoming a single <UNK>
    token."""
    corpus = {"low": 5, "lowest": 2, "newer": 6, "wider": 3, "new": 2}
    # "</w>" marks a word boundary so "er" inside "newer" and a trailing "er" don't
    # get confused with an "er" that continues into the next character -- standard
    # BPE convention (NOTE-ML-4 evidence item 2 describes the merge algorithm; the
    # end-of-word marker is the standard bookkeeping detail every BPE write-up uses
    # to keep merges from crossing word boundaries).
    vocab: dict[tuple[str, ...], int] = {tuple(w) + ("</w>",): f for w, f in corpus.items()}

    print("\n--- BPE tokenizer demo ---")
    print(f"toy corpus (word: frequency): {corpus}")
    print(f"initial vocab (each word split into characters): {vocab}")

    learned_merges: list[tuple[str, str]] = []
    for step in range(1, n_merges + 1):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)  # type: ignore[arg-type]
        vocab = merge_vocab(best_pair, vocab)
        learned_merges.append(best_pair)
        print(f"merge {step}: {best_pair} (count={pairs[best_pair]}) -> "
              f"{best_pair[0] + best_pair[1]!r}")

    print(f"final vocab after {len(learned_merges)} merges: {vocab}")
    subword_symbols = sorted({sym for word in vocab for sym in word})
    print(f"subword symbols learned: {subword_symbols}")

    print("\napplying the learned merges to words NOT in the training corpus:")
    for unseen_word in ["slower", "wildest"]:
        tokens = apply_bpe(unseen_word, learned_merges)
        print(f"  {unseen_word!r:12s} -> {tokens}")

    return learned_merges


# ---------------------------------------------------------------------------
# 2. Embeddings: real sentence-transformer, with a deterministic hand-made fallback
# ---------------------------------------------------------------------------

def hand_made_embeddings(words: list[str], dim: int = 9, seed: int = RNG_SEED) -> np.ndarray:
    """Deterministic, dependency-free stand-in for real embeddings, used only if
    sentence-transformers (or its model download) is unavailable. 7 axes are chosen
    by hand to encode exactly the relationships this chapter wants to demonstrate --
    royalty, gender, "seniority" (adult vs junior royal), and four unrelated
    categories -- plus 2 dims of small fixed-seed noise so vectors aren't perfectly
    axis-aligned. Because "king - man + woman" and "queen" share the same
    (royal=1, gender=+1, generation=1) coordinates by construction, the analogy demo
    below holds exactly the same way it would need real training data to hold for a
    real word2vec model. NOTE-ML-4's own recommendation: "numpy hand-vectors ...
    sufficient to demonstrate cosine/Euclidean concepts" (evidence item 6)."""
    axes = {
        #                royal gender  generation  fruit vehicle capital country
        "king":     [1, -1, 1, 0, 0, 0, 0],
        "queen":    [1, 1, 1, 0, 0, 0, 0],
        "man":      [0, -1, 1, 0, 0, 0, 0],
        "woman":    [0, 1, 1, 0, 0, 0, 0],
        "prince":   [1, -1, 0.3, 0, 0, 0, 0],
        "princess": [1, 1, 0.3, 0, 0, 0, 0],
        "apple":    [0, 0, 0, 1, 0, 0, 0],
        "orange":   [0, 0, 0, 1, 0, 0, 0],
        "car":      [0, 0, 0, 0, 1, 0, 0],
        "truck":    [0, 0, 0, 0, 1, 0, 0],
        "paris":    [0, 0, 0, 0, 0, 1, 0],
        "france":   [0, 0, 0, 0, 0, 0, 1],
        "london":   [0, 0, 0, 0, 0, 1, 0],
        "england":  [0, 0, 0, 0, 0, 0, 1],
    }
    rng = np.random.default_rng(seed)
    rows = []
    for w in words:
        base = np.array(axes[w], dtype=np.float64)
        noise = rng.normal(0.0, 0.04, size=dim - len(base))
        rows.append(np.concatenate([base, noise]))
    V = np.array(rows)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def get_embeddings(words: list[str]) -> tuple[np.ndarray, str]:
    """Tries a real sentence-transformer model first; falls back to
    `hand_made_embeddings()` on any failure (no internet, model not cached, import
    error, etc.), printing which path was taken either way. Returns L2-normalized
    embeddings (unit vectors) either way, so the d^2 = 2*(1-cos) identity holds in
    both cases."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        V = model.encode(words, normalize_embeddings=True).astype(np.float64)
        source = (
            f"sentence-transformers/all-MiniLM-L6-v2 (real embeddings, {V.shape[1]}-D, "
            f"Apache 2.0 -- NOTE-ML-4 evidence item 6)"
        )
        print(f"[embeddings] loaded {source}")
        return V, source
    except Exception as exc:  # pragma: no cover - exercised only when offline/uncached
        print(f"[embeddings] could not load sentence-transformers model ({type(exc).__name__}: {exc})")
        print("[embeddings] falling back to hand-made 9-D vectors (see hand_made_embeddings()).")
        V = hand_made_embeddings(words)
        source = f"hand-made fallback vectors ({V.shape[1]}-D, deterministic, seed={RNG_SEED})"
        return V, source


# ---------------------------------------------------------------------------
# 3. Analogy geometry: king - man + woman ~= queen
# ---------------------------------------------------------------------------

def run_analogy_demo(V: np.ndarray, idx: dict[str, int]) -> None:
    """LO2: word2vec-style analogy geometry. Computes king - man + woman, re-normalizes
    it to a unit vector, and ranks every other word in the vocabulary by cosine
    similarity to that combined vector. NOTE-ML-4 evidence item 3: "king" - "man" +
    "woman" ~= "queen"."""
    analogy = V[idx["king"]] - V[idx["man"]] + V[idx["woman"]]
    analogy = analogy / np.linalg.norm(analogy)

    candidates = [w for w in idx if w not in ("king", "man", "woman")]
    sims = {w: float(analogy @ V[idx[w]]) for w in candidates}
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])

    print("\n--- analogy demo: king - man + woman ~= ? ---")
    for word, sim in ranked[:5]:
        print(f"  cos(analogy, {word:10s}) = {sim:.4f}")
    top_word, top_sim = ranked[0]
    assert top_word == "queen", (
        f"expected 'queen' as the nearest neighbour of king-man+woman, got {top_word!r} "
        f"(cos={top_sim:.4f}) -- embedding source may have changed"
    )
    print(f"nearest neighbour: {top_word!r} (cos={top_sim:.4f}) -- matches the expected geometry.")


# ---------------------------------------------------------------------------
# 4. Cosine similarity vs Euclidean distance, on real vectors
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """cos_sim(a, b) = (a . b) / (||a|| ||b||). NOTE-ML-4 evidence item 1."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """d(a, b) = sqrt(sum((a_i - b_i)^2)). NOTE-ML-4 evidence item 1."""
    return float(np.linalg.norm(a - b))


def build_cosine_vs_euclidean_table(V: np.ndarray, idx: dict[str, int], path: Path) -> None:
    """Computes cosine similarity and Euclidean distance for several word pairs on the
    SAME (unit-norm) vectors, verifies the normalization identity
    d^2(a,b) = 2*(1 - cos_sim(a,b)) numerically for each pair, then demonstrates the
    "Euclidean is magnitude-sensitive, cosine isn't" pitfall by comparing a vector to
    a rescaled copy of itself. Renders everything as a table image (LO3 + the spec's
    "cosine-vs-Euclidean comparison table" artefact) and saves it."""
    pairs = [
        ("king", "queen"), ("king", "man"), ("apple", "orange"),
        ("car", "truck"), ("king", "apple"), ("paris", "france"), ("paris", "london"),
    ]

    print("\n--- cosine vs Euclidean, on the same unit-norm vectors ---")
    rows = []
    for w1, w2 in pairs:
        a, b = V[idx[w1]], V[idx[w2]]
        cos = cosine_sim(a, b)
        euc = euclidean_dist(a, b)
        predicted_euc_sq = 2.0 * (1.0 - cos)
        assert abs(euc**2 - predicted_euc_sq) < 1e-6, (
            f"normalization identity failed for ({w1}, {w2}): "
            f"euc^2={euc**2:.6f} vs 2*(1-cos)={predicted_euc_sq:.6f}"
        )
        rows.append((f"{w1} / {w2}", f"{cos:.4f}", f"{euc:.4f}", f"{euc**2:.4f}", f"{predicted_euc_sq:.4f}"))
        print(f"  {w1:8s} vs {w2:8s}  cos={cos: .4f}  euclidean={euc:.4f}  "
              f"euclidean^2={euc**2:.4f}  2*(1-cos)={predicted_euc_sq:.4f}  (match)")

    # --- magnitude-sensitivity pitfall: same direction, 3x the length ---
    king = V[idx["king"]]
    king_scaled = king * 3.0
    cos_scaled = cosine_sim(king, king_scaled)
    euc_scaled = euclidean_dist(king, king_scaled)
    pitfall_row = ("king / king*3 (pitfall)", f"{cos_scaled:.4f}", f"{euc_scaled:.4f}", "n/a", "n/a (not unit-norm)")
    print(f"\n  PITFALL -- king vs king scaled by 3x (same direction, different magnitude):")
    print(f"  cos={cos_scaled:.4f} (unchanged -- cosine ignores magnitude)   "
          f"euclidean={euc_scaled:.4f} (grows with the scale factor)")

    col_labels = ["pair", "cosine sim.", "Euclidean d.", "d^2", "2*(1-cos)"]
    table_rows = rows + [pitfall_row]

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * (len(table_rows) + 1) + 0.6))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    # Highlight the pitfall row so the reader's eye lands on it.
    for col in range(len(col_labels)):
        table[(len(table_rows), col)].set_facecolor("#fde8d0")
    ax.set_title(
        "Cosine similarity vs Euclidean distance on unit-norm word embeddings\n"
        "(all pairs satisfy d^2 = 2*(1 - cos); last row: rescaling breaks Euclidean, not cosine)",
        fontsize=10, pad=14,
    )
    fig.tight_layout()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 5. 2-D projection of the embeddings (PCA via SVD on centred data)
# ---------------------------------------------------------------------------

def plot_embedding_scatter_2d(V: np.ndarray, words: list[str], source_label: str, path: Path) -> None:
    """Projects the embedding matrix down to 2 dimensions with PCA -- computed here as
    a plain SVD on mean-centred data (no extra library: PCA's principal components are
    exactly the right singular vectors of the centred data matrix). This is for
    VISUALIZATION only, not for the cosine/Euclidean numbers above, which are always
    computed on the full-dimensional vectors -- 2D necessarily throws away most of the
    variance, which is exactly why similarity search is never done in a PCA-reduced
    space in practice."""
    centred = V - V.mean(axis=0, keepdims=True)
    _, singular_values, Vt = np.linalg.svd(centred, full_matrices=False)
    coords_2d = centred @ Vt[:2].T
    explained = (singular_values[:2] ** 2).sum() / (singular_values**2).sum()

    # A couple of points land very close together (e.g. "paris"/"france" are highly
    # similar in this embedding space -- itself a nice illustration of the geometry).
    # Nudge their labels apart so the text stays readable.
    label_offset = {"paris": (6, -10), "france": (6, 10)}

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for word, (x, y) in zip(words, coords_2d):
        color = CATEGORY_COLOR[CATEGORY[word]]
        ax.scatter(x, y, color=color, s=60, zorder=3)
        xytext = label_offset.get(word, (6, 4))
        ax.annotate(word, (x, y), textcoords="offset points", xytext=xytext, fontsize=9)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=cat)
        for cat, c in CATEGORY_COLOR.items()
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, title="category")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(
        f"2-D PCA projection of word embeddings ({source_label.split(' (')[0]})\n"
        f"top-2 components explain {explained * 100:.1f}% of variance -- a 2-D picture, not the real geometry",
        fontsize=10,
    )
    fig.tight_layout()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"top-2 principal components explain {explained * 100:.1f}% of total variance "
          f"(the rest is lost in this 2-D picture).")


# ---------------------------------------------------------------------------
# 6. Quantization: fp32 -> fp16 -> int8, measured on the real embedding matrix
# ---------------------------------------------------------------------------

def quantization_demo(V: np.ndarray) -> None:
    """Casts the embedding matrix through the three precision levels NOTE-ML-4
    evidence item 4 describes (fp32 -> fp16 -> int8) and measures, on THIS data: bytes
    used and how much cosine similarity survives the round trip. int8 here is a
    hand-rolled symmetric min-max quantizer (scale by the max absolute value, round to
    the nearest of 256 integer levels, scale back) -- the simplest form of the
    post-training quantization NOTE-ML-4 describes, not a claim about any specific
    library's calibration algorithm."""
    fp32 = V.astype(np.float32)
    fp16 = fp32.astype(np.float16)

    scale = float(np.abs(fp32).max())
    int8_codes = np.round(fp32 / scale * 127).astype(np.int8)
    int8_reconstructed = (int8_codes.astype(np.float32) / 127.0) * scale

    def kib(arr: np.ndarray) -> float:
        return arr.nbytes / 1024.0

    def mean_cosine_preserved(original: np.ndarray, reconstructed: np.ndarray) -> float:
        sims = [cosine_sim(original[i], reconstructed[i]) for i in range(original.shape[0])]
        return float(np.mean(sims))

    fp16_cos = mean_cosine_preserved(fp32, fp16.astype(np.float32))
    int8_cos = mean_cosine_preserved(fp32, int8_reconstructed)

    print(f"\n--- quantization on the {fp32.shape[0]}x{fp32.shape[1]} embedding matrix ---")
    print(f"  fp32: {kib(fp32):6.2f} KiB  (baseline, 4 bytes/value)")
    print(f"  fp16: {kib(fp16):6.2f} KiB  (2 bytes/value, {kib(fp32) / kib(fp16):.1f}x smaller)  "
          f"mean cosine vs fp32 = {fp16_cos:.6f}")
    print(f"  int8: {kib(int8_codes):6.2f} KiB  (1 byte/value, {kib(fp32) / kib(int8_codes):.1f}x smaller)  "
          f"mean cosine vs fp32 = {int8_cos:.6f}")
    print("  (fp16 and int8 sizes/ratios follow directly from the byte width per NOTE-ML-4 evidence item 4; "
          "the cosine-preserved numbers are measured on this run's own vectors, not a claimed benchmark.)")


# ---------------------------------------------------------------------------
def main() -> None:
    run_bpe_demo()

    V, source_label = get_embeddings(WORDS)
    idx = {w: i for i, w in enumerate(WORDS)}

    run_analogy_demo(V, idx)
    build_cosine_vs_euclidean_table(V, idx, ARTEFACTS_DIR / "cosine_vs_euclidean_table.png")
    plot_embedding_scatter_2d(V, WORDS, source_label, ARTEFACTS_DIR / "embedding_scatter_2d.png")
    quantization_demo(V)


if __name__ == "__main__":
    main()
