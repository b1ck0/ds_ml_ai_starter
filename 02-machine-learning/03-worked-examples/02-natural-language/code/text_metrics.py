"""SPEC-ML-14 worked example: text & NLP metrics for classification, generation, retrieval.

Every metric below is computed TWICE: once as a plain-Python/NumPy "by hand" implementation
of the formula (grounded in research/NOTE-ML-14-metric-definitions.md), and once via the
pinned library that ships the same metric in production. The two are printed side by side so
you can see they agree — library output is not magic, it is the same arithmetic, faster.

Sections:
  1. Classification  -- macro / micro / weighted F1            (by hand + scikit-learn)
  2. Generation       -- perplexity, BLEU, chrF, ROUGE, BERTScore (by hand + library, ONE
                         candidate/reference pair used throughout, so you can see the three
                         generation metrics DISAGREE on the very same pair)
  3. Retrieval        -- cosine similarity, Recall@k, MRR, nDCG  (by hand + library where one
                         exists)

Environment: .venv-ml (Python 3.13, CPU only). Pinned per NOTE-ML-14-package-versions.md:
  sacrebleu==2.6.0  rouge-score==0.1.2  bert-score==0.3.13  scikit-learn==1.9.0
  evaluate==0.4.6   torch==2.14.0+cpu   transformers==5.16.1
Run: .venv-ml/Scripts/python.exe text_metrics.py
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

# GPT-2's tokenizer prints byte-level markers (e.g. "Ġ" for a leading space) that the
# default Windows console codepage (cp1252) cannot encode -- force UTF-8 stdout so token
# printouts below don't crash on Windows. Harmless on Linux/macOS (already UTF-8).
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF
from sklearn.metrics import f1_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed

SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
set_seed(SEED)
torch.manual_seed(SEED)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Classification metrics — macro / micro / weighted F1
# ---------------------------------------------------------------------------

# A tiny, deliberately imbalanced 3-class example (classes: A=7 true, B=2 true, C=1 true).
# [source: NOTE-ML-14-metric-definitions.md — F1 = harmonic mean of precision and recall,
# same definition SPEC-DS-6 used for the binary case, applied per class.]
CLS_LABELS = ["A", "A", "A", "A", "A", "A", "A", "B", "B", "C"]
CLS_PREDS = ["A", "A", "A", "A", "A", "A", "B", "B", "C", "C"]
CLS_CLASSES = ["A", "B", "C"]


def f1_per_class_by_hand(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict[str, dict]:
    """Per-class precision/recall/F1, implemented directly from the confusion counts.

    precision_c = TP_c / (TP_c + FP_c); recall_c = TP_c / (TP_c + FN_c);
    F1_c = 2 * precision_c * recall_c / (precision_c + recall_c).
    """
    results = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[c] = {"tp": tp, "fp": fp, "fn": fn, "support": support,
                      "precision": precision, "recall": recall, "f1": f1}
    return results


def macro_micro_weighted_f1_by_hand(per_class: dict[str, dict], n_total: int) -> dict[str, float]:
    """Aggregate per-class F1 three ways.

    macro    = plain mean of per-class F1 (every class counts equally, regardless of size)
    micro    = pool TP/FP/FN across ALL classes first, then compute one F1 — for single-label
               multiclass this collapses to accuracy (verified below)
    weighted = mean of per-class F1, weighted by each class's support (true count)
    [source: NOTE-ML-14-metric-definitions.md — F1 aggregation, tied to SPEC-DS-6 precision/recall.]
    """
    macro = sum(v["f1"] for v in per_class.values()) / len(per_class)
    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())
    total_fn = sum(v["fn"] for v in per_class.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
             if (micro_precision + micro_recall) > 0 else 0.0)
    weighted = sum(v["f1"] * v["support"] for v in per_class.values()) / n_total
    return {"macro_f1": macro, "micro_f1": micro, "weighted_f1": weighted}


def section_1_classification() -> dict:
    banner("1. Classification metrics -- macro / micro / weighted F1")
    per_class = f1_per_class_by_hand(CLS_LABELS, CLS_PREDS, CLS_CLASSES)
    for c, v in per_class.items():
        print(f"  class {c}: TP={v['tp']} FP={v['fp']} FN={v['fn']} support={v['support']}  "
              f"precision={v['precision']:.4f} recall={v['recall']:.4f} f1={v['f1']:.4f}")
    hand = macro_micro_weighted_f1_by_hand(per_class, len(CLS_LABELS))
    accuracy = sum(1 for t, p in zip(CLS_LABELS, CLS_PREDS) if t == p) / len(CLS_LABELS)
    print(f"\n  [by hand]     macro_f1={hand['macro_f1']:.4f}  micro_f1={hand['micro_f1']:.4f}  "
          f"weighted_f1={hand['weighted_f1']:.4f}  accuracy={accuracy:.4f}")

    sk_macro = f1_score(CLS_LABELS, CLS_PREDS, labels=CLS_CLASSES, average="macro")
    sk_micro = f1_score(CLS_LABELS, CLS_PREDS, labels=CLS_CLASSES, average="micro")
    sk_weighted = f1_score(CLS_LABELS, CLS_PREDS, labels=CLS_CLASSES, average="weighted")
    print(f"  [scikit-learn] macro_f1={sk_macro:.4f}  micro_f1={sk_micro:.4f}  "
          f"weighted_f1={sk_weighted:.4f}")
    return {"hand": hand, "sklearn": {"macro_f1": sk_macro, "micro_f1": sk_micro,
                                       "weighted_f1": sk_weighted}, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# 2. Generation metrics — perplexity, BLEU, chrF, ROUGE, BERTScore
# ---------------------------------------------------------------------------

PERPLEXITY_MODEL_ID = "distilbert/distilgpt2"  # same causal LM as SPEC-ML-9, Apache-2.0
PERPLEXITY_SENTENCE = "The cat sat on the mat."

# The ONE candidate/reference pair used for every generation metric below: a genuine
# paraphrase that shares almost no words with the reference. This is the cold-open example,
# made concrete -- watch BLEU, chrF, and ROUGE score it low (surface overlap is small) while
# BERTScore scores it high (the meaning is close).
REFERENCE = "the cat sat on the mat"
CANDIDATE = "a feline rested on the rug"

BERTSCORE_MODEL_ID = "distilbert-base-uncased"  # NOT the bert-score default (roberta-large,
# ~1.4GB); this is the same small encoder SPEC-ML-13 already fine-tuned, so it is fast and
# CPU-friendly here. bert_score.score()'s `model_type=` override is documented in
# NOTE-ML-14-library-apis.md. Production use should keep the roberta-large default for the
# best correlation with human judgement (same NOTE).


def section_2a_perplexity() -> dict:
    banner("2a. Perplexity -- by hand vs. the model's own loss")
    tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL_ID)
    model.eval()

    encoded = tokenizer(PERPLEXITY_SENTENCE, return_tensors="pt")
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"  sentence: {PERPLEXITY_SENTENCE!r}")
    print(f"  tokens ({len(tokens)}): {tokens}")

    with torch.no_grad():
        logits = model(**encoded).logits  # (1, seq_len, vocab_size)

    # By hand: perplexity = exp(mean negative log-likelihood per token)
    # [source: NOTE-ML-14-metric-definitions.md, item 4 -- HF transformers perplexity docs].
    # Each token (from the 2nd onward) is scored by the model's predicted distribution over
    # the PREVIOUS tokens -- shift logits/labels by one position, exactly like next-token
    # prediction during training.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)
    mean_nll_by_hand = -token_log_probs.mean().item()
    perplexity_by_hand = math.exp(mean_nll_by_hand)
    print(f"  [by hand]      mean NLL/token = {mean_nll_by_hand:.4f}  ->  perplexity = "
          f"{perplexity_by_hand:.4f}")

    # Library: transformers' causal-LM loss IS mean cross-entropy per token when you pass
    # labels= -- the same shift-by-one is done internally.
    with torch.no_grad():
        library_loss = model(**encoded, labels=input_ids).loss.item()
    perplexity_library = math.exp(library_loss)
    print(f"  [transformers] mean NLL/token = {library_loss:.4f}  ->  perplexity = "
          f"{perplexity_library:.4f}")
    return {"mean_nll_by_hand": mean_nll_by_hand, "perplexity_by_hand": perplexity_by_hand,
            "mean_nll_library": library_loss, "perplexity_library": perplexity_library}


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(candidate_tokens: list[str], reference_tokens: list[str], n: int) -> tuple[float, int, int]:
    """Clipped n-gram precision at order n: min(count_in_candidate, count_in_reference) summed,
    divided by the candidate's total n-gram count. [source: NOTE-ML-14-metric-definitions.md,
    item 1 -- Papineni et al. (2002).]
    """
    cand_ngrams = ngrams(candidate_tokens, n)
    ref_ngrams = ngrams(reference_tokens, n)
    if not cand_ngrams:
        return 0.0, 0, 0
    clipped = sum(min(count, ref_ngrams[g]) for g, count in cand_ngrams.items())
    total = sum(cand_ngrams.values())
    return (clipped / total if total > 0 else 0.0), clipped, total


def bleu_by_hand(candidate: str, reference: str, max_n: int = 4) -> dict:
    cand_tokens, ref_tokens = candidate.split(), reference.split()
    precisions = [modified_precision(cand_tokens, ref_tokens, n) for n in range(1, max_n + 1)]
    c, r = len(cand_tokens), len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    p_values = [p for p, _, _ in precisions]
    if min(p_values) > 0:
        geo_mean = math.exp(sum(math.log(p) for p in p_values) / max_n)
        raw_bleu = bp * geo_mean
    else:
        raw_bleu = 0.0  # a zero at any order kills the unsmoothed geometric mean
    return {"precisions": precisions, "brevity_penalty": bp, "raw_bleu": raw_bleu,
            "cand_len": c, "ref_len": r}


def chrf_by_hand(candidate: str, reference: str, char_order: int = 2, beta: int = 2) -> dict:
    """A simplified chrF: character n-grams up to `char_order`, computed over the
    whitespace-stripped strings (chrF ignores word boundaries by design -- that is the whole
    point of a *character* n-gram metric).
    chrF_beta = (1+beta^2) * P * R / (beta^2 * P + R)
    [source: NOTE-ML-14-metric-definitions.md, item 2 -- Popovic (2015).]
    We use char_order=2 (not sacrebleu's higher default order) so the n-gram counts below are
    small enough to sanity-check by eye; the library call is configured to match exactly.

    IMPORTANT (found by comparing against sacrebleu, see prose below): precision/recall are
    computed PER n-gram order first, and only THEN averaged across orders -- pooling raw
    counts across orders before dividing gives a different (wrong) number. This is exactly
    what "verify by actually running it" catches that reading the formula alone would not.
    """
    cand_chars = candidate.replace(" ", "")
    ref_chars = reference.replace(" ", "")
    order_precisions, order_recalls = [], []
    for n in range(1, char_order + 1):
        cand_ng = Counter(cand_chars[i:i + n] for i in range(len(cand_chars) - n + 1))
        ref_ng = Counter(ref_chars[i:i + n] for i in range(len(ref_chars) - n + 1))
        matched = sum(min(cnt, ref_ng[g]) for g, cnt in cand_ng.items())
        cand_total, ref_total = sum(cand_ng.values()), sum(ref_ng.values())
        order_precisions.append(matched / cand_total if cand_total > 0 else 0.0)
        order_recalls.append(matched / ref_total if ref_total > 0 else 0.0)
    precision = sum(order_precisions) / len(order_precisions)
    recall = sum(order_recalls) / len(order_recalls)
    denom = beta ** 2 * precision + recall
    chrf = (1 + beta ** 2) * precision * recall / denom if denom > 0 else 0.0
    return {"precision": precision, "recall": recall, "chrf": chrf,
            "order_precisions": order_precisions, "order_recalls": order_recalls}


def rouge_by_hand(candidate: str, reference: str) -> dict:
    """ROUGE-1 (unigram recall/precision), ROUGE-2 (bigram), ROUGE-L (LCS), all with the
    harmonic-mean F-measure. [source: NOTE-ML-14-metric-definitions.md, item 3 -- Lin (2004).]
    """
    cand_tokens, ref_tokens = candidate.split(), reference.split()

    def rouge_n(n: int) -> dict:
        cand_ng, ref_ng = ngrams(cand_tokens, n), ngrams(ref_tokens, n)
        matched = sum(min(cnt, ref_ng[g]) for g, cnt in cand_ng.items())
        cand_total, ref_total = sum(cand_ng.values()), sum(ref_ng.values())
        precision = matched / cand_total if cand_total > 0 else 0.0
        recall = matched / ref_total if ref_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "fmeasure": f1}

    def lcs_length(a: list[str], b: list[str]) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision_l = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall_l = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1_l = 2 * precision_l * recall_l / (precision_l + recall_l) if (precision_l + recall_l) > 0 else 0.0

    return {"rouge1": rouge_n(1), "rouge2": rouge_n(2),
            "rougeL": {"precision": precision_l, "recall": recall_l, "fmeasure": f1_l, "lcs": lcs}}


def bertscore_by_hand(candidate: str, reference: str, tokenizer, model) -> dict:
    """Token-level BERTScore precision/recall/F1, implemented directly from contextual
    embeddings of `model` (special tokens [CLS]/[SEP] excluded, last hidden layer):
    precision = mean over candidate tokens of (max cosine similarity to any reference token)
    recall    = mean over reference tokens of (max cosine similarity to any candidate token)
    F1        = harmonic mean(precision, recall)
    [source: NOTE-ML-14-metric-definitions.md, item 5 -- Zhang et al. (2020).]
    """
    def token_embeddings(text: str) -> torch.Tensor:
        enc = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state[0]  # (seq_len, hidden_dim)
        # drop [CLS] (index 0) and [SEP] (last index) -- BERTScore only scores content tokens
        return hidden[1:-1]

    cand_emb = token_embeddings(candidate)
    ref_emb = token_embeddings(reference)
    cand_norm = torch.nn.functional.normalize(cand_emb, dim=-1)
    ref_norm = torch.nn.functional.normalize(ref_emb, dim=-1)
    sim = cand_norm @ ref_norm.T  # (n_cand_tokens, n_ref_tokens) cosine similarity matrix

    precision = sim.max(dim=1).values.mean().item()
    recall = sim.max(dim=0).values.mean().item()
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "similarity_matrix": sim.numpy()}


def section_2b_generation() -> dict:
    banner("2b. Generation metrics -- BLEU, chrF, ROUGE, BERTScore on ONE pair")
    print(f"  reference: {REFERENCE!r}")
    print(f"  candidate: {CANDIDATE!r}   (a paraphrase -- almost no shared words)")

    # --- BLEU ---
    print("\n  --- BLEU ---")
    hand_bleu = bleu_by_hand(CANDIDATE, REFERENCE)
    for n, (p, clipped, total) in enumerate(hand_bleu["precisions"], start=1):
        print(f"    [by hand] P{n} = {clipped}/{total} = {p:.4f}")
    print(f"    [by hand] brevity_penalty={hand_bleu['brevity_penalty']:.4f}  "
          f"raw (unsmoothed) BLEU = {hand_bleu['raw_bleu']:.4f}  "
          f"(a zero at any n-gram order zeroes the whole geometric mean)")

    bleu_none = BLEU(effective_order=True, smooth_method="none")
    score_none = bleu_none.sentence_score(CANDIDATE, [REFERENCE])
    bleu_exp = BLEU(effective_order=True, smooth_method="exp")
    score_exp = bleu_exp.sentence_score(CANDIDATE, [REFERENCE])
    print(f"    [sacrebleu, smooth_method='none'] score = {score_none.score:.4f}  "
          f"(0-100 scale; matches the by-hand 0 above at *100)")
    print(f"    [sacrebleu, smooth_method='exp']  score = {score_exp.score:.4f}  "
          f"(default smoothing avoids the hard zero -- this is the number you'd actually report)")

    # --- chrF ---
    print("\n  --- chrF ---")
    hand_chrf = chrf_by_hand(CANDIDATE, REFERENCE, char_order=2, beta=2)
    print(f"    [by hand] per-order precision={['%.4f' % p for p in hand_chrf['order_precisions']]}  "
          f"per-order recall={['%.4f' % r for r in hand_chrf['order_recalls']]}")
    print(f"    [by hand, char_order=2] averaged precision={hand_chrf['precision']:.4f}  "
          f"recall={hand_chrf['recall']:.4f}  chrF={hand_chrf['chrf']:.4f}")
    chrf_metric = CHRF(char_order=2, word_order=0, beta=2)
    chrf_score = chrf_metric.sentence_score(CANDIDATE, [REFERENCE])
    print(f"    [sacrebleu CHRF(char_order=2, word_order=0, beta=2)] score = "
          f"{chrf_score.score:.4f}  (0-100 scale)")
    chrf_default = CHRF()
    chrf_default_score = chrf_default.sentence_score(CANDIDATE, [REFERENCE])
    print(f"    [sacrebleu CHRF(), library defaults]                 score = "
          f"{chrf_default_score.score:.4f}  (0-100 scale, higher char order -- not hand-matched)")

    # --- ROUGE ---
    print("\n  --- ROUGE ---")
    hand_rouge = rouge_by_hand(CANDIDATE, REFERENCE)
    for name, v in hand_rouge.items():
        print(f"    [by hand] {name}: precision={v['precision']:.4f} recall={v['recall']:.4f} "
              f"fmeasure={v['fmeasure']:.4f}")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    lib_rouge = scorer.score(target=REFERENCE, prediction=CANDIDATE)
    for name, v in lib_rouge.items():
        print(f"    [rouge-score] {name}: precision={v.precision:.4f} recall={v.recall:.4f} "
              f"fmeasure={v.fmeasure:.4f}")

    # --- BERTScore ---
    print("\n  --- BERTScore ---")
    bs_tokenizer = AutoTokenizer.from_pretrained(BERTSCORE_MODEL_ID)
    bs_model = AutoModel.from_pretrained(BERTSCORE_MODEL_ID)
    bs_model.eval()
    hand_bs = bertscore_by_hand(CANDIDATE, REFERENCE, bs_tokenizer, bs_model)
    print(f"    [by hand, {BERTSCORE_MODEL_ID}] precision={hand_bs['precision']:.4f}  "
          f"recall={hand_bs['recall']:.4f}  F1={hand_bs['f1']:.4f}")

    from bert_score import score as bert_score_fn
    # Pitfall, found by actually running this (not documented behaviour we assumed):
    # `num_layers=None` is supposed to mean "use the last layer" (NOTE-ML-14-library-apis.md).
    # That is true for models bert-score has a TUNED default for (its internal WMT16-correlation
    # lookup table). distilbert-base-uncased is not one of them: num_layers=None silently falls
    # back to a DIFFERENT layer (5 of 6), giving F1=0.8729, not the by-hand F1=0.7991 above,
    # which uses the true last hidden layer. Passing num_layers=6 explicitly reproduces it exactly.
    P_default, R_default, F1_default = bert_score_fn(
        [CANDIDATE], [REFERENCE], model_type=BERTSCORE_MODEL_ID, num_layers=None, verbose=False)
    print(f"    [bert-score, num_layers=None (\"last layer\" per docs)] F1={F1_default.item():.4f}  "
          f"<- does NOT match the by-hand last-layer F1 above")
    P, R, F1 = bert_score_fn([CANDIDATE], [REFERENCE], model_type=BERTSCORE_MODEL_ID,
                              num_layers=6, verbose=False)
    print(f"    [bert-score, num_layers=6 (the TRUE last layer)]    precision={P.item():.4f}  "
          f"recall={R.item():.4f}  F1={F1.item():.4f}  <- matches by hand exactly")

    return {
        "bleu": {"hand_raw": hand_bleu["raw_bleu"], "sacrebleu_none": score_none.score / 100,
                 "sacrebleu_exp": score_exp.score / 100},
        "chrf": {"hand": hand_chrf["chrf"] / 100 if hand_chrf["chrf"] > 1 else hand_chrf["chrf"],
                 "sacrebleu_matched": chrf_score.score / 100},
        "rouge1_f": lib_rouge["rouge1"].fmeasure,
        "rouge2_f": lib_rouge["rouge2"].fmeasure,
        "rougeL_f": lib_rouge["rougeL"].fmeasure,
        "bertscore_f1_hand": hand_bs["f1"],
        "bertscore_f1_library": F1.item(),          # num_layers=6, matches hand exactly
        "bertscore_f1_library_default": F1_default.item(),  # num_layers=None, the out-of-the-box call
    }


# ---------------------------------------------------------------------------
# 3. Retrieval metrics — cosine similarity, Recall@k, MRR, nDCG
# ---------------------------------------------------------------------------

RETRIEVAL_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # same embedding model SPEC-AGENT-3 uses
QUERY_TEXT = "How do I reset my password?"
DOC_TEXTS = ["Password reset instructions for your account.", "Our lunch menu changes weekly."]

# Three toy queries' ranked results (already sorted by the retriever's score, rank 1 first).
# relevance: binary relevant/not-relevant per ranked slot; total_relevant: how many relevant
# docs exist in the WHOLE corpus for that query (can exceed what's visible in the top-k).
RETRIEVAL_QUERIES = [
    {"name": "Q1", "relevance": [0, 1, 0, 1, 1], "total_relevant": 4},
    {"name": "Q2", "relevance": [1, 0, 0, 0, 0], "total_relevant": 1},
    {"name": "Q3", "relevance": [0, 0, 0, 0, 1], "total_relevant": 2},
]
# A fourth, graded-relevance ranked list for nDCG (0=irrelevant .. 3=highly relevant).
GRADED_RELEVANCE = [0, 2, 0, 1, 3]


def recall_at_k(relevance: list[int], total_relevant: int, k: int) -> float:
    """Recall@k = (# relevant docs found in the top-k) / (total # relevant docs).
    [source: NOTE-ML-14-metric-definitions.md, item 7.]"""
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def mrr(rr_ranks: list[int]) -> float:
    """MRR = mean(1 / rank_of_first_correct_result), 1-indexed ranks.
    [source: NOTE-ML-14-metric-definitions.md, item 8.]"""
    return sum(1.0 / r for r in rr_ranks) / len(rr_ranks)


def first_relevant_rank(relevance: list[int]) -> int:
    for i, rel in enumerate(relevance, start=1):
        if rel > 0:
            return i
    return 0  # no relevant doc found at all -> undefined; excluded from MRR in that case


def ndcg_by_hand(graded_relevance: list[int], k: int | None = None, gain: str = "exp") -> dict:
    """nDCG@k = DCG@k / IDCG@k, log2(rank+1) discount.
    [source: NOTE-ML-14-metric-definitions.md, item 9 -- Wikipedia "Discounted cumulative gain".]

    `gain="exp"` uses exponential gain (2^rel - 1) -- the formula most search/RAG literature
    means by "nDCG" (it weights highly-relevant results much more heavily), and the one
    NOTE-ML-14-metric-definitions.md quotes. `gain="linear"` uses the older gain = rel_i
    directly, from the same Wikipedia article's original (Jarvelin & Kekalainen) formulation.
    Both are real, named variants of the SAME metric -- which one a library implements is not
    always obvious from its name alone (see the scikit-learn comparison below).
    """
    k = k or len(graded_relevance)

    def dcg(relevances: list[int]) -> float:
        gains = [(2 ** rel - 1) if gain == "exp" else rel for rel in relevances[:k]]
        return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

    actual_dcg = dcg(graded_relevance)
    ideal_dcg = dcg(sorted(graded_relevance, reverse=True))
    return {"dcg": actual_dcg, "idcg": ideal_dcg,
            "ndcg": actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0}


def section_3_retrieval() -> dict:
    banner("3. Retrieval metrics -- cosine similarity, Recall@k, MRR, nDCG")

    # --- cosine similarity ---
    print("  --- Cosine similarity ---")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(RETRIEVAL_MODEL_ID)
    query_vec = embed_model.encode([QUERY_TEXT])[0]
    doc_vecs = embed_model.encode(DOC_TEXTS)

    def cosine_by_hand(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    hand_sims = [cosine_by_hand(query_vec, d) for d in doc_vecs]
    sk_sims = sk_cosine_similarity([query_vec], doc_vecs)[0]
    for i, (doc, h, s) in enumerate(zip(DOC_TEXTS, hand_sims, sk_sims)):
        print(f"    doc {i} ({doc[:40]!r}...): [by hand]={h:.4f}  [sklearn]={s:.4f}")

    # --- Recall@k ---
    print("\n  --- Recall@k (k=1, 3, 5) ---")
    recall_rows = []
    for q in RETRIEVAL_QUERIES:
        row = {"query": q["name"]}
        for k in (1, 3, 5):
            row[f"recall@{k}"] = recall_at_k(q["relevance"], q["total_relevant"], k)
        recall_rows.append(row)
        print(f"    {q['name']} (relevance={q['relevance']}, total_relevant={q['total_relevant']}): "
              + "  ".join(f"R@{k}={row[f'recall@{k}']:.4f}" for k in (1, 3, 5)))
    for k in (1, 5):
        avg = sum(r[f"recall@{k}"] for r in recall_rows) / len(recall_rows)
        print(f"    mean Recall@{k} across queries = {avg:.4f}")

    # --- MRR ---
    print("\n  --- MRR ---")
    ranks = [first_relevant_rank(q["relevance"]) for q in RETRIEVAL_QUERIES]
    for q, r in zip(RETRIEVAL_QUERIES, ranks):
        print(f"    {q['name']}: first relevant result at rank {r}  ->  RR = {1 / r:.4f}")
    hand_mrr = mrr(ranks)
    print(f"    [by hand] MRR = {hand_mrr:.4f}")
    print("    (no dedicated scikit-learn Recall@k / MRR function exists -- these are"
          " retrieval-specific and normally come from a library like pytrec_eval/ranx;"
          " the by-hand implementation above IS the reference here.)")

    # --- nDCG ---
    print("\n  --- nDCG (graded relevance) ---")
    print(f"    graded relevance, in rank order: {GRADED_RELEVANCE}")
    hand_ndcg_exp = ndcg_by_hand(GRADED_RELEVANCE, gain="exp")
    print(f"    [by hand, exponential gain 2^rel-1] DCG={hand_ndcg_exp['dcg']:.4f}  "
          f"IDCG={hand_ndcg_exp['idcg']:.4f}  nDCG={hand_ndcg_exp['ndcg']:.4f}")
    hand_ndcg_lin = ndcg_by_hand(GRADED_RELEVANCE, gain="linear")
    print(f"    [by hand, linear gain = rel]        DCG={hand_ndcg_lin['dcg']:.4f}  "
          f"IDCG={hand_ndcg_lin['idcg']:.4f}  nDCG={hand_ndcg_lin['ndcg']:.4f}")
    # sklearn.metrics.ndcg_score expects 2D arrays: y_true=relevance, y_score=values whose
    # ORDER reproduces the ranking being evaluated (descending score = the given rank order).
    y_true = np.asarray([GRADED_RELEVANCE])
    y_score = np.asarray([list(range(len(GRADED_RELEVANCE), 0, -1))])  # 5,4,3,2,1 -> same order
    sk_ndcg = ndcg_score(y_true, y_score)
    print(f"    [sklearn.metrics.ndcg_score]        nDCG = {sk_ndcg:.4f}  "
          f"<- matches the LINEAR-gain hand computation, not the exponential-gain one")

    return {"cosine_hand": hand_sims, "cosine_sklearn": sk_sims.tolist(),
            "recall_rows": recall_rows, "mrr_hand": hand_mrr,
            "ndcg_hand_exp": hand_ndcg_exp["ndcg"], "ndcg_hand_linear": hand_ndcg_lin["ndcg"],
            "ndcg_sklearn": float(sk_ndcg)}


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def write_artefacts(gen_results: dict) -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        ("BLEU (smoothed)", gen_results["bleu"]["sacrebleu_exp"]),
        ("chrF (matched order)", gen_results["chrf"]["sacrebleu_matched"]),
        ("ROUGE-1 F1", gen_results["rouge1_f"]),
        ("ROUGE-2 F1", gen_results["rouge2_f"]),
        ("ROUGE-L F1", gen_results["rougeL_f"]),
        ("BERTScore F1", gen_results["bertscore_f1_library_default"]),
    ]

    csv_path = ARTEFACTS_DIR / "textmetrics_generation_disagreement.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "score_0_to_1", "reference", "candidate"])
        for name, value in rows:
            writer.writerow([name, f"{value:.4f}", REFERENCE, CANDIDATE])
    print(f"\n  wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#4c72b0"] * 5 + ["#c44e52"]  # BERTScore stands out from the n-gram metrics
    ax.barh(names, values, color=colors)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("score (0-1 scale)")
    ax.set_title(f"Same pair, disagreeing scores\nref: {REFERENCE!r}\ncandidate (paraphrase): {CANDIDATE!r}",
                 fontsize=11)
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    png_path = ARTEFACTS_DIR / "textmetrics_generation_disagreement.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {png_path}")


def main() -> None:
    cls_results = section_1_classification()
    ppl_results = section_2a_perplexity()
    gen_results = section_2b_generation()
    ret_results = section_3_retrieval()
    write_artefacts(gen_results)

    banner("Summary: hand vs. library, every metric")
    print(f"  classification macro-F1:  hand={cls_results['hand']['macro_f1']:.4f}  "
          f"sklearn={cls_results['sklearn']['macro_f1']:.4f}")
    print(f"  perplexity:                hand={ppl_results['perplexity_by_hand']:.4f}  "
          f"transformers={ppl_results['perplexity_library']:.4f}")
    print(f"  BLEU (smooth='none'):      hand={gen_results['bleu']['hand_raw']:.4f}  "
          f"sacrebleu={gen_results['bleu']['sacrebleu_none']:.4f}")
    print(f"  BERTScore F1:              hand={gen_results['bertscore_f1_hand']:.4f}  "
          f"bert-score(num_layers=6)={gen_results['bertscore_f1_library']:.4f}")
    print(f"  nDCG (exponential gain):   hand={ret_results['ndcg_hand_exp']:.4f}  "
          f"sklearn={ret_results['ndcg_sklearn']:.4f}  (deliberately DON'T match -- different gain fn)")
    print(f"  nDCG (linear gain):        hand={ret_results['ndcg_hand_linear']:.4f}  "
          f"sklearn={ret_results['ndcg_sklearn']:.4f}  (match)")


if __name__ == "__main__":
    main()
