# NOTE-ML-14: Current library APIs for NLP metrics (sacrebleu, rouge-score, bert-score)

**Date checked:** 2026-09-03  
**Versions verified:** sacrebleu 2.6.0, rouge-score 0.1.2, bert-score 0.3.13

## Answer
Exact API signatures and calling conventions for the three main metric libraries, verified against source code and documentation.

---

## 1. SacreBLEU (sacrebleu 2.6.0)

### High-level API (object-oriented, recommended)

```python
from sacrebleu.metrics import BLEU

bleu = BLEU()
score = bleu.corpus_score(sys, refs)  # sys: list of str, refs: list of list of str
# Returns BLEUScore with .score attribute (0–100 scale) and .signature
```

### Lower-level API (backward compatibility, ≤ 1.4.10 style)

```python
from sacrebleu import corpus_bleu, sentence_bleu

# Corpus-level (full test set)
corpus_bleu(hypotheses, references, smooth_method='exp', smooth_value=None,
            force=False, lowercase=False, tokenize='13a', use_effective_order=False)
# hypotheses: Sequence[str] (one per test example)
# references: Sequence[Sequence[str]] (multiple refs per example)
# Returns: BLEUScore with .score (0–100)

# Sentence-level (per-example)
sentence_bleu(hypothesis, references)  # hypothesis: str, references: Sequence[str]
# Note: As of sacrebleu 2.0+, now uses exponential smoothing (matches CLI)
# Returns: BLEUScore with .score (0–100)
```

### Key differences
- **Corpus-level BLEU:** Accumulates counts across all sentences, then computes precision once (standard).
- **Sentence-level BLEU:** Computes BLEU per sentence (smoothing needed to avoid zero penalty when a 4-gram is unseen).
- **Default smoothing:** Changed in 2.0+; `sentence_bleu()` now uses exponential smoothing to match CLI behavior.

### Tokenization (defaults to '13a')
```python
bleu = BLEU(tokenize='13a')  # mteval-v13a (default)
# Other options: 'zh' (Chinese), 'ja' (Japanese), 'ko' (Korean), 'char', 'intl' (language auto-detect)
```

**Source:** [SacreBLEU GitHub](https://github.com/mjpost/sacrebleu/); [sacrebleu/sacrebleu/compat.py](https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/compat.py); [blog: "Computing and reporting BLEU scores"](https://bricksdont.github.io/posts/2020/12/computing-and-reporting-bleu-scores/)

---

## 2. rouge-score (rouge_score 0.1.2)

### Main API

```python
from rouge_score import rouge_scorer

# Create scorer for one or more ROUGE variants
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

# Score a single hypothesis-reference pair
scores = scorer.score(target, prediction)
# target: str (reference)
# prediction: str (candidate)
# Returns: dict {
#   'rouge1': Score(precision=..., recall=..., fmeasure=...),
#   'rouge2': Score(precision=..., recall=..., fmeasure=...),
#   'rougeL': Score(precision=..., recall=..., fmeasure=...),
# }
# Each Score is a namedtuple with .precision, .recall, .fmeasure (all in [0, 1])
```

### Options
```python
RougeScorer(
    rouge_types=['rouge1', 'rouge2', 'rougeL'],  # Options: 'rouge1', 'rouge2', 'rougeL', 'rougeW'
    use_stemmer=False,  # If True, Porter stemmer applied (use_stemmer=True matches NIST ROUGE-1.5.5)
    split_summaries=False
)
```

### Interpretation
- `rouge1`: Unigram overlap (recall of 1-grams)
- `rouge2`: Bigram overlap (recall of 2-grams)
- `rougeL`: LCS (recall of longest common subsequence)
- All return precision, recall, and F-measure (harmonic mean)

**Source:** [rouge_score GitHub](https://github.com/google-research/google-research/tree/master/rouge); PyPI package description: "Pure python implementation of ROUGE-1.5.5"

**Note:** This is a pure Python port; for exact NIST ROUGE-1.5.5 scoring, use `use_stemmer=True`.

---

## 3. BERTScore (bert-score 0.3.13)

### Main API

```python
from bert_score import score

P, R, F1 = score(
    cands,  # List[str] or Iterable[str]: candidate/hypothesis texts
    refs,   # List[str] or Iterable[str]: reference texts (one per candidate)
    lang='en',  # Language: 'en', 'zh', 'tr', etc.
    model_type='roberta-large',  # Optional: override default model
    num_layers=None,  # Number of layers to use (if None, uses last layer)
    all_layers=False,  # If True, concatenate all layer embeddings
    batch_size=64,  # GPU batch size
    num_threads=4,  # CPU threads for tokenization
    device='cuda:0' if torch.cuda.is_available() else 'cpu',  # Device
    max_length=512,  # Max token length (truncates longer sequences)
    rescale_with_baseline=False,  # Apply baseline rescaling to [0, 1]
    baseline_path=None,  # Path to custom baseline CSV/TSV
    baseline_url=None,  # URL to custom baseline CSV/TSV
    idf=False,  # If True, weight tokens by inverse document frequency
    verbose=False,  # Print progress
    return_hash=False  # If True, return ((P, R, F1), hash_string)
)
# Returns: (P, R, F1) where each is a tensor of shape (N,), N = # of examples
# If return_hash=True: ((P, R, F1), hash_code) for reproducibility logging
```

### Default models by language
- **English ('en'):** `roberta-large`
- **Chinese ('zh'):** `bert-base-chinese`
- **Turkish ('tr'):** `dbmdz/bert-base-turkish-cased`
- **Other languages:** `bert-base-multilingual-cased`

### Recommended model for higher human correlation
For improved alignment with human judgment: `microsoft/deberta-xlarge-mnli` (instead of `roberta-large`).

### Rescale-with-baseline explanation
If `rescale_with_baseline=True`, scores are rescaled via: `score_rescaled = (score - baseline) / (1 - baseline)`, where baseline is the average score over ~500k random candidate-reference pairs. This stretches the narrow [0.85, 0.95] range to [0, 1] for interpretability, **without affecting correlation with human judgment**.

**Source:** [Tiiiger/bert_score GitHub](https://github.com/Tiiiger/bert_score); [Zhang et al. (2020) BERTScore paper](https://arxiv.org/pdf/1904.09675); [rescale_baseline.md](https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md)

---

## Caveats & limits

1. **SacreBLEU:**
   - Sentence-level BLEU changed in version 2.0+ (now uses exponential smoothing). Ensure chapter tests against 2.6.0 to match behavior.
   - Tokenizer '13a' is the standard for MT evaluation; document if using language-specific tokenizers.

2. **rouge-score:**
   - Unmaintained library (last release 2022-07-22); no newer version on PyPI.
   - Pure Python implementation; slower than Perl ROUGE-1.5.5 but matches it with `use_stemmer=True`.
   - API is per-pair scoring; no built-in corpus averaging (compute manually or use `evaluate` library wrapper).

3. **BERTScore:**
   - Model download is lazy (first call fetches and caches). Large model size (~400 MB for roberta-large).
   - GPU memory required: ~8 GB for batch_size=64 with roberta-large.
   - Token-level scoring; truncates sequences at max_length=512 (may lose information in long documents).
   - Rescale-with-baseline is optional but recommended for interpretability; does **not** change correlation with humans.

## Recommendation

**For the chapter code:**

1. **BLEU:** Use the object-oriented API for clarity:
   ```python
   from sacrebleu.metrics import BLEU
   bleu = BLEU(tokenize='13a')
   score = bleu.corpus_score(hypotheses, references)
   print(f"BLEU = {score.score:.2f}")
   ```

2. **ROUGE:** Iterate over examples and aggregate manually or use the `evaluate` library wrapper:
   ```python
   from rouge_score import rouge_scorer
   scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
   scores = [scorer.score(ref, hyp) for hyp, ref in zip(candidates, references)]
   avg_rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
   ```

3. **BERTScore:** Use with explicit language and model:
   ```python
   from bert_score import score
   P, R, F1 = score(candidates, references, lang='en', model_type='roberta-large')
   print(f"BERTScore: P={P.mean():.4f}, R={R.mean():.4f}, F1={F1.mean():.4f}")
   ```

4. **Pin versions** in requirements.txt or pyproject.toml (see NOTE-ML-14-package-versions).

5. **Document tokenization and model choices** in code comments and output so snippets are reproducible.
