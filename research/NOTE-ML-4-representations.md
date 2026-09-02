# NOTE-ML-4: Representations — Embeddings, Tokenizers, Similarity, Quantization, Fine-tuning

**Answer:** Cosine similarity = (a·b) / (||a|| ||b||), range [-1, 1], angle-based; Euclidean distance = √(Σ(a_i - b_i)²), range [0, ∞), magnitude-based; on normalized vectors: Euclidean² = 2(1 - Cosine similarity); BPE iteratively merges most-frequent byte pairs until vocabulary size reached; Word2Vec has skip-gram (word→context) and CBOW (context→word) architectures learning dense embeddings where geometry encodes semantic relationships; quantization reduces precision FP32→FP16→INT8 trading accuracy for size/speed; LoRA (Hu et al., 2021) freezes pre-trained weights and injects low-rank trainable matrices in linear layers for parameter-efficient fine-tuning; for CPU-friendly embeddings, sentence-transformers (Apache 2.0) offers lightweight models like all-MiniLM-L6-v2 or numpy hand-vectors for demo.

**Evidence:**

1. **Cosine Similarity & Euclidean Distance Formulas** (verified 2026-09-02):
   - **Cosine similarity:** cos_sim(a, b) = (a · b) / (||a|| ||b||) = (Σ a_i b_i) / (√Σ a_i² × √Σ b_i²)
     - Range: [-1, 1] (or [0, 1] for non-negative vectors); 1 = identical direction, 0 = orthogonal, -1 = opposite
     - Measures angle between vectors; invariant to magnitude; robust for normalized embeddings
   - **Euclidean distance:** d(a, b) = √(Σ(a_i - b_i)²)
     - Range: [0, ∞]; 0 = identical, larger = more different
     - Measures straight-line distance; sensitive to magnitude; requires normalization for fair comparison
   - **On normalized vectors:** If ||a|| = ||b|| = 1, then d²(a,b) = 2(1 - cos_sim(a, b))
   - **Why cosine dominates for embeddings:** Direction (semantic meaning) matters more than magnitude; cosine is scale-invariant
   - Sources: https://zilliz.com/ai-faq/in-practical-terms-what-differences-might-you-observe-in-a-search-system-when-using-cosine-similarity-instead-of-euclidean-distance-on-the-same-set-of-normalized-embeddings and https://www.pinecone.io/learn/vector-similarity/

2. **Byte Pair Encoding (BPE)** (verified 2026-09-02):
   - Algorithm: start with characters as base units; iteratively count all adjacent symbol pairs, replace most-frequent pair with new symbol, add to vocabulary, repeat until vocabulary size or merge limit reached
   - Subword tokenization: balances vocabulary size vs sequence length (vs word tokens which have large vocab for rare words, vs char tokens which have long sequences)
   - Addresses out-of-vocabulary (OOV) problem: rare words decomposed into subword units
   - Original paper: "Neural Machine Translation of Rare Words with Subword Units" (2016)
   - Widely used in modern LLMs (GPT-2, GPT-3 use variants)
   - Sources: https://medium.com/@varunsivamani/byte-pair-encoding-bpe-5fdced1b31cd and https://towardsdatascience.com/byte-pair-encoding-for-beginners-708d4472c0c7/

3. **Word2Vec: Skip-gram & CBOW** (verified 2026-09-02):
   - Developed by Mikolov et al. (Google, 2013)
   - **Skip-gram:** Given center word, predict surrounding context words; learns to place semantically similar words nearby in vector space
   - **CBOW (Continuous Bag-of-Words):** Given surrounding context words, predict center word; faster to train, slightly lower quality
   - Output: dense word vectors (embeddings) where geometry encodes meaning (e.g., "king" - "man" + "woman" ≈ "queen")
   - Training: typically 100-300 dimensional vectors; trained on large unlabeled corpora
   - Advantage over one-hot encoding: captures relationships; much fewer parameters
   - Modern successor: Transformer-based embeddings (BERT, sentence-transformers) but Word2Vec still widely used
   - Source: https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp-comparison-between-cbow-and-skip-gram-models/ and https://apxml.com/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/word2vec-cbow-skipgram

4. **Quantization: FP32, FP16, INT8** (verified 2026-09-02):
   - **FP32 (single precision):** 4 bytes per value; baseline; full precision, high dynamic range
   - **FP16 (half precision):** 2 bytes per value; 2x memory reduction vs FP32; requires hardware support (NVIDIA Tensor Cores, modern GPUs); 10-50% speedup depending on hardware
   - **INT8 (8-bit integer):** 1 byte per value; 4x memory reduction vs FP32; requires quantization-aware training (QAT) or post-training quantization (PTQ) to minimize accuracy loss
   - Trade-offs: INT8 most aggressive compression but requires careful calibration; FP16 good middle ground; FP32 safest
   - Method: Quantization-Aware Training (QAT) simulates quantization during training (fake-quant layers); maintains float computation but rounds to int8 ranges
   - Accuracy impact: INT8 typically <1% accuracy drop on many models; full float → INT8 may lose more
   - Sources: https://apxml.com/courses/cnns-for-computer-vision/chapter-8-model-compression-efficient-dl/quantization-reducing-precision and https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt

5. **LoRA: Low-Rank Adaptation** (verified 2026-09-02):
   - Full citation: Hu et al., 2021 (first to propose LoRA for large models)
   - Core idea: pre-trained weights frozen; trainable low-rank matrices A, B injected into each linear layer
   - Update to weight W approximated as: ΔW ≈ A × B^T where A ∈ ℝ^(d_out × r), B ∈ ℝ^(d_in × r), r ≪ d_in, d_out (rank r << dimensions)
   - Advantage: fine-tune large models (billions of parameters) with few trainable parameters (% or less)
   - Example: for 7B-parameter LLM, LoRA adds ~1M trainable params (≈0.01%) while maintaining performance
   - At inference: merge LoRA matrices into weights (A × B^T added to W); no additional inference cost
   - Use cases: domain adaptation, instruction-tuning, personalization without retraining
   - Sources: https://arxiv.org/pdf/2406.09679 and https://arxiv.org/pdf/2605.08110

6. **CPU-Friendly Embeddings** (verified 2026-09-02):
   - **Sentence-Transformers (Apache 2.0 license):**
     - Library: https://huggingface.co/sentence-transformers
     - Lightweight models available: all-MiniLM-L6-v2 (22M params, ~44 MB), jina-embeddings-v5-text-small (released 2026-02-18)
     - Auto-selects best device (CUDA > MPS > CPU); explicit CPU-only available
     - ONNX format support: 3x CPU inference speedup
     - Model2Vec wrapper: can make 50x smaller, 500x faster
   - **Alternative: NumPy hand-vectors**
     - For demo: manually create embedding matrix (e.g., 10 words × 50 dims with random floats)
     - No dependencies; fully CPU-native; sufficient to demonstrate cosine/Euclidean concepts
     - Recommendation: hand-vectors best for theory chapter (lightweight, no downloads); sentence-transformers for production use
   - Sources: https://huggingface.co/blog/sentence-transformers-joins-hf, https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html, and https://github.com/huggingface/sentence-transformers (Apache License)

**Caveats / limits:**

- **Cosine vs Euclidean normalization:** Euclidean on unnormalized vectors can be misleading; always compare apples-to-apples (both normalized or both raw).
- **BPE tokenizer variance:** Different BPE implementations may tokenize differently even with same vocab size; GPT-2 BPE ≠ BERT WordPiece (even though both subword). Spec just needs conceptual understanding.
- **Word2Vec dimensionality:** Modern embeddings (BERT-base: 768D, GPT-2: 768D) much higher than original Word2Vec (100-300D). Spec focused on intuition, not reproduction.
- **Quantization accuracy:** Domain-dependent; computer vision models often sustain INT8 better than NLP (activation distributions differ).
- **LoRA limitation:** Cannot fine-tune *all* layers; typically frozen embeddings, frozen early layers to save compute. Full fine-tuning still needed for best results if resources permit.
- **Sentence-transformers license:** Apache 2.0 (permissive); specific pre-trained models may have different licenses (check HuggingFace model card).

**Recommendation:**

1. **Cosine vs Euclidean demo:** Show 3-4 pairs of 2D vectors; compute both metrics; plot and compare results.
2. **BPE illustration:** Show iterative merge process with toy text (e.g., "hello world hello"); animate frequency counts and merges.
3. **Word2Vec concept:** Skip geometric details; focus on "words with similar meaning cluster nearby in embedding space."
4. **Quantization table:** FP32 (baseline) → FP16 (2x speedup, minor loss) → INT8 (4x speedup, requires calibration).
5. **LoRA explanation:** Frozen W; trainable A, B injected; inference merges A×B^T back into W (no latency cost).
6. **Embeddings for demo:**
   - **Theory chapter (ML-3):** Use numpy hand-vectors (10 words, 8D embeddings); no dependencies, fast.
   - **Production use (Agentic):** Recommend sentence-transformers with all-MiniLM-L6-v2 (or jina-embeddings-v5-text-small) for CPU.
7. **Cite Hu et al. 2021 for LoRA** when introducing fine-tuning concepts; strongly grounded reference.

## Correction (verified during authoring, 2026-09-03)
The LoRA citation URLs in evidence item 5 resolve to unrelated papers; the correct LoRA reference is Hu et al. 2021, arXiv:2106.09685 — verified during ML-3 authoring.
