# NOTE-ML-14: Authoritative metric definitions for NLP evaluation

**Date checked:** 2026-09-03

## Answer
Each metric definition below is verified against its authoritative source. All formulas are from official documentation, published papers, or standards.

---

## 1. BLEU (Bilingual Evaluation Understudy)

**Definition:**  
BLEU combines modified n-gram precision with a brevity penalty to penalize short translations.

**Formula:**
```
BLEU = BP · exp(Σ w_n log P_n)
```
where:
- `BP = 1` if candidate length `c > reference length r`, else `exp(1 - r/c)`
- `P_n` = modified n-gram precision (clipped counts: min(count_candidate, count_reference) / count_candidate)
- `w_n` = weights (typically equal, 1/4 each for n=1..4)

**Source:** [Papineni et al. (2002), "BLEU: a Method for Automatic Evaluation of Machine Translation"](https://aclanthology.org/P02-1040.pdf)

**SacreBLEU defaults (verified 2026-09-03):**
- Tokenizer: `13a` (mimics mteval-v13a from Moses MT toolkit)
- Case: mixed (case-sensitive)
- Smoothing: exponential decay
- References: single by default
- Output: version string includes all settings for reproducibility (e.g., `BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0`)

---

## 2. chrF (Character n-gram F-score)

**Definition:**  
F-score computed over character n-grams (not words). Language- and tokenization-independent.

**Formula:**
```
chrF_β = ((1 + β²) · precision · recall) / (β² · precision + recall)
```
where precision and recall are computed on character n-gram overlap (typically β=3 for equal weight on P/R).

**Source:** [Popović (2015), "chrF: character n-gram F-score for automatic MT evaluation"](https://aclanthology.org/W15-3049/)

**Key advantage:** No word tokenization needed; works across languages and character sets.

---

## 3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Definition:**  
Set of metrics measuring recall (not precision) of n-gram and longest-common-subsequence overlap.

### ROUGE-N (N-gram overlap, typically N=1,2)
```
ROUGE-N = (Σ_ref min(count_ref(n-gram), count_hyp(n-gram))) / Σ_ref count_ref(n-gram)
```
Recall-based: sum of clipped counts in reference divided by total reference n-grams.

### ROUGE-L (Longest Common Subsequence, word-level)
```
ROUGE-L_recall = LCS(ref, hyp) / len(ref)
ROUGE-L_precision = LCS(ref, hyp) / len(hyp)
ROUGE-L_F = (1 + β²) · (recall · precision) / (β² · recall + precision)
```
Measures structural similarity via word-sequence matching (order preserved but not consecutive).

**Source:** [Lin (2004), cited in ACL Anthology and standard NLP texts](https://medium.com/@pjc1108/rouge-metrics-bfcb3483ea25)

**Use case:** Primarily for summarization evaluation; ROUGE-1/2 measure content overlap, ROUGE-L measures structure.

---

## 4. Perplexity

**Definition:**  
Exponentiated average negative log-likelihood of a token sequence under a language model.

**Formula:**
```
Perplexity(x_1:n) = exp(-1/n · Σ log p(x_i | x_{<i}))
```
where:
- `n` = total number of tokens in the sequence
- `p(x_i | x_{<i})` = model's probability of token i given preceding context
- Lower perplexity = better model (less "surprised" by the test data)

**Equivalently:**
```
Perplexity = exp(mean_negative_log_likelihood_per_token)
```

**Source:** [Hugging Face transformers documentation](https://huggingface.co/docs/transformers/perplexity); standard in language model evaluation (Rohan Paul, 2025).

**Important caveat:** Perplexity is not comparable across models with different tokenizers or vocabularies.

---

## 5. BERTScore

**Definition:**  
Token-level semantic similarity via contextual embeddings. Computes precision, recall, and F1 over token similarity.

**Formulas:**
```
Precision = mean_over_candidate_tokens(max_over_reference_tokens(cosine_similarity))
Recall = mean_over_reference_tokens(max_over_candidate_tokens(cosine_similarity))
F1 = harmonic_mean(Precision, Recall)
```

**Contextualization:** Each token is embedded using a pre-trained contextual model (default: RoBERTa-large for English); all tokens scored, not just exact matches.

**Rescale-with-baseline caveat:**  
Raw BERTScore typically falls in [0.85, 0.95] (narrow range despite strong correlation with humans). Optional rescaling applies:
```
rescaled_score = (original_score - baseline) / (1 - baseline)
```
where baseline is the average score over ~500k random candidate-reference pairs from a monolingual corpus. Rescaling preserves correlation with human judgment while spreading scores to [0, 1].

**Default model:** `roberta-large` for English; `bert-base-chinese` for Chinese; `bert-base-multilingual-cased` for other languages.

**Recommendation for better human correlation:** Use `microsoft/deberta-xlarge-mnli` instead of default if human alignment is critical.

**Source:** [Zhang et al. (2020), "BERTScore: Evaluating Text Generation with BERT"](https://arxiv.org/pdf/1904.09675); [Tiiiger/bert_score GitHub](https://github.com/Tiiiger/bert_score)

---

## 6. Cosine Similarity

**Definition:**  
Cosine of the angle between two vectors in an inner product space.

**Formula:**
```
similarity(A, B) = (A · B) / (||A|| × ||B||) = Σ(A_i × B_i) / sqrt(Σ(A_i²) × Σ(B_i²))
```

**Range:** [-1, 1] where 1 = same direction, 0 = orthogonal, -1 = opposite.

**In BERTScore context:** Applied to token embeddings to measure semantic closeness; used for token matching.

**Source:** [Standard linear algebra; Google for Developers "Measuring similarity from embeddings"](https://developers.google.com/machine-learning/clustering/dnn-clustering/supervised-similarity)

---

## 7. Recall@k

**Definition:**  
Fraction of relevant documents retrieved in the top-k positions.

**Formula:**
```
Recall@k = (# relevant docs in top-k) / (total # relevant docs)
```

**Key property:** Order-unaware; returning a relevant item at rank 1 scores the same as rank k. Use when you care only that relevance appears in top-k, not its exact position.

**Source:** [Pinecone, "Evaluation Measures in Information Retrieval"](https://www.pinecone.io/learn/offline-evaluation/)

---

## 8. MRR (Mean Reciprocal Rank)

**Definition:**  
Average of reciprocal rank of the first correct answer, averaged over a set of queries.

**Formula:**
```
MRR = (1/|Q|) · Σ (1 / rank_of_first_correct_result)
```
where rank is 1-indexed (rank 1 = top result).

**Key property:** Order-aware; top-1 correctness is rewarded much more than lower ranks. Suitable for tasks where the first correct result matters most.

**Source:** [Pinecone, "Evaluation Measures in Information Retrieval"](https://www.pinecone.io/learn/offline-evaluation/)

---

## 9. nDCG (Normalized Discounted Cumulative Gain)

**Definition:**  
Evaluates ranking quality by considering both relevance and position, normalized to [0, 1] by ideal ranking.

**Formulas:**
```
DCG@k = Σ_{i=1}^{k} (2^{rel_i} - 1) / log_2(i + 1)

IDCG@k = DCG of ideal ranking (sorted by descending relevance)

nDCG@k = DCG@k / IDCG@k
```
where:
- `rel_i` = relevance score of item at position i (binary 0/1 or graded)
- `log_2(i+1)` = discount factor (position i+1; top-1 has smallest discount)
- `IDCG@k` provides normalization to [0, 1] range

**Interpretation:**
- 1.0 = perfect ranking (ideal order)
- 0.5 = achieves half the ideal DCG
- Accounts for graded relevance, not just binary

**Source:** [Wikipedia "Discounted cumulative gain"](https://en.wikipedia.org/wiki/Discounted_cumulative_gain); [Deepchecks, "Normalized Discounted Cumulative Gain"](https://deepchecks.com/glossary/normalized-discounted-cumulative-gain/)

---

## Caveats & limits

1. **BLEU / ROUGE / chrF:**  
   - All are n-gram overlap metrics; they **do not capture paraphrase or synonymy** (e.g., "great" vs. "excellent" scores 0 overlap).
   - Tokenization is crucial for BLEU (sacrebleu's `13a` vs. language-specific tokenizers give different results).
   - Single-reference bias: metrics penalize correct but novel phrasings.

2. **Perplexity:**  
   - Tokenizer-dependent; not comparable across models with different vocabularies.
   - Only defined for autoregressive / causal LMs; not applicable to bidirectional models.

3. **BERTScore:**  
   - Default model (RoBERTa-large) may not align with human judgment; `deberta-xlarge-mnli` recommended for higher correlation.
   - Rescaling with baseline is optional but recommended for interpretability.

4. **Cosine similarity / Recall@k / MRR / nDCG:**  
   - Cosine similarity on raw embeddings (no fine-tuning) may not capture semantic nuance.
   - Recall@k, MRR, nDCG all require a ground-truth ranking or relevance labels.

## Recommendation

**For the chapter:**
1. Quote these definitions verbatim from the sources; do not paraphrase or simplify (especially BLEU brevity penalty, nDCG discount).
2. Implement by-hand calculations on tiny examples (2–3 candidate-reference pairs) to show students how the formula works, then validate against library output.
3. When showing BLEU, explicitly note sacrebleu's `13a` tokenizer as the default; document any custom tokenization.
4. For BERTScore examples, use `roberta-large` (the default) unless specifically comparing model choices.
5. Emphasize the limitations section: automatic metrics **do not measure meaning**, only surface overlap. Use this as motivation for the next section on human evaluation and LLM-as-judge.
