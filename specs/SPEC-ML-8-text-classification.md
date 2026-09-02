# SPEC-ML-8: Text Classification with Transformers (RoBERTa/DistilBERT)

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Worked Examples (Natural Language)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-3 (embeddings/tokenizers), SPEC-DS-6 (classification metrics)
**Nature:** RUNNABLE ON CPU with a SMALL model — use a compact pretrained encoder for inference and a
tiny fine-tune (or zero-shot/head-only) so it runs without a GPU.

## Intent
Apply a transformer encoder to a real NLP task: sentiment/topic classification. Show the HuggingFace
workflow (tokenizer → model → logits → label) and connect back to the classification metrics from DS-6.

## Learning objectives
- LO1 — Explain why a pretrained encoder (RoBERTa/DistilBERT/BERT) beats bag-of-words for text.
- LO2 — Use the HuggingFace `transformers` pipeline/tokenizer/model to classify text and read the logits.
- LO3 — Evaluate on a small labelled set with the DS-6 metrics (accuracy, F1, confusion matrix).
- LO4 — Understand the two adaptation paths: use a task-specific pretrained head vs fine-tune on your data.

## Scope
In: tokenizer + pretrained encoder inference on a small dataset; metric evaluation; a small/optional head-only fine-tune that runs on CPU.
Out: full-scale fine-tuning (conceptual + link), distributed training.

## Outline
1. What & why — transformers vs classical NLP; transfer learning for text.
2. Tokenize → encode → classify with a small pretrained model (e.g. distilbert-sst2).
3. Evaluate on a small labelled set — accuracy/F1/confusion matrix.
4. Adaptation — pretrained task head vs fine-tuning (keep any training tiny/CPU-friendly).
5. Pitfalls — tokenizer/model mismatch, sequence length/truncation, label mapping.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/natural-language/text-classification.md"
- Code: "Machine Learning/Worked Examples/natural-language/code/text_classification.py"
- Artefacts: predictions table; confusion matrix on the small eval set.

## Claims to ground (Haiku, before writing)
- [ ] Verify current `transformers` (+ tokenizers, and torch) versions and the current pipeline/AutoTokenizer/AutoModelForSequenceClassification API. Confirm a SMALL CPU-friendly model id that downloads freely (e.g. distilbert-base-uncased-finetuned-sst-2-english) and its licence.
- [ ] Confirm a tiny labelled dataset that's easy to load (a few hand-written examples, or a small slice of a datasets-hub set) — recommend the runnable path.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — text_classification.py RUNS on CPU with a real small model, classifies + evaluates, produces artefacts; snippet-check passes; no fabricated predictions. AC3 — transformers version + model id + APIs grounded. AC4 — ties to DS-6 metrics; transfer-learning intuition clear.

## Gates
Entry: approved; notes landed. Exit: DoD checklist. Uses .venv-ml (+ transformers).
