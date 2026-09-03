# SPEC-ML-13: Fine-tuning a transformer — training a real model end to end

**Status:** approved
**Subject:** Machine Learning
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-3 (Representations — embeddings, tokenizers, fine-tuning concept),
SPEC-ML-8 (Text classification — inference on a pretrained encoder), SPEC-DS-4 (train/valid/holdout),
SPEC-DS-6 (classification metrics)

## Intent
The reader has, so far, only *run* pretrained models: ML-4 trains a small CNN from scratch, but every
NLP chapter (ML-8, ML-9) is inference-only, and ML-8 §4 *describes* "two adaptation paths" without
executing either. This chapter closes that gap: it fine-tunes a real pretrained encoder
(DistilBERT) on a small labelled text dataset, end to end — load data, tokenize, train, watch the
loss fall and validation accuracy climb epoch by epoch, evaluate, save, reload, and run inference on
new text. This is the "closer to perfection with each iteration" story made concrete for a
transformer. A Java dev has trained models conceptually; here they see the Python/PyTorch/HF training
loop and the `Trainer` abstraction that wraps it, with a JVM analogy (the loop as a framework
lifecycle you configure rather than hand-write).

## Learning objectives
After this chapter the reader can:
- LO1 — Explain what fine-tuning changes vs. training from scratch (start from pretrained weights,
  adapt the head + optionally the body; far less data and compute), and when each is right.
- LO2 — Load a labelled dataset (HF `datasets`), tokenize it correctly (padding/truncation, the
  tokenizer that matches the checkpoint), and build train/validation splits without leakage.
- LO3 — Run a full fine-tuning loop with the HF `Trainer` (and understand the equivalent explicit
  PyTorch loop it replaces: forward → loss → backward → optimizer step → zero grad), reading the
  per-epoch training loss and validation metric as they improve.
- LO4 — Evaluate the fine-tuned model against the pretrained baseline on held-out data, save the
  model + tokenizer to disk, reload them, and run inference on unseen text.
- LO5 — Name the common fine-tuning pitfalls (learning rate too high → catastrophic forgetting,
  too few/too many epochs, train/eval tokenizer mismatch, class imbalance, evaluating on train).

## Scope
In scope: supervised fine-tuning of a small encoder for sequence classification on a small dataset,
on CPU in minutes (small subset + few epochs) with a note on GPU; the `Trainer` API and the explicit
loop it wraps; before/after comparison; save/load/inference.
Out of scope: parameter-efficient fine-tuning (LoRA/PEFT), full-model instruction tuning of decoders,
RLHF/DPO, distributed/multi-GPU training (link ML-12 cloud GPU chapter). A one-paragraph "what's next:
LoRA" pointer is enough.

## Outline (section-by-section)
1. **Cold open** — a short, grounded origin note on transfer learning / fine-tuning (ULMFiT 2018 and
   BERT 2018–19 made "pretrain then fine-tune" the default NLP recipe). Then the problem: ML-8's
   pretrained sentiment head is generic; our labels aren't its labels. We can't run inference on a
   task the model was never trained for. Pose it, let the reader feel it.
2. **What & why** — fine-tuning vs. from-scratch, with the Java analogy; the "you are here" map
   (representations → inference → **fine-tuning** → metrics).
3. **The data** — load the dataset, inspect real rows, tokenize (show the tokenizer output for one
   real sentence), split. Real numbers: dataset size, label distribution, sequence lengths.
4. **The model & the loop** — load the pretrained checkpoint for classification (fresh head), show
   the explicit PyTorch loop once (forward/loss/backward/step/zero-grad) so the abstraction isn't
   magic, then run it via `Trainer`. Watch training loss ↓ and validation accuracy ↑ per epoch
   (iterate-visibly: print/plot the curve).
5. **Evaluate & compare** — fine-tuned vs. pretrained-baseline on the held-out set (accuracy + macro-F1,
   confusion matrix artefact). Save model+tokenizer, reload, infer on new sentences.
6. **Pitfalls** — LR, epochs, tokenizer mismatch, leakage, imbalance, forgetting.
7. **Recap & next** — pointer to SPEC-ML-14 (text metrics) and a one-line LoRA teaser.

## Assets to produce
- Prose: `02-machine-learning/03-worked-examples/02-natural-language/03-fine-tuning-a-transformer.md`
- Code: `02-machine-learning/03-worked-examples/02-natural-language/code/fine_tune_classifier.py`
  (runnable end to end on CPU with a small subset; seed set; deps pinned to the NOTE versions)
- Artefacts: a training-curve plot (loss + val accuracy per epoch) and a confusion matrix, both
  reproduced by the code, under `.../02-natural-language/artefacts/`
- A short saved-model note (where it writes, how to reload) — the model dir itself is gitignored.

## Claims to ground (Haiku research brief — do BEFORE writing)
- [ ] Package versions to pin: `transformers`, `datasets`, `torch`, `evaluate`, `scikit-learn`,
      `accelerate` (Trainer dependency) — current stable versions on PyPI, with dates.
- [ ] Dataset URL + licence: primary candidate `dair-ai/emotion` (HF Hub) — confirm it loads via
      `datasets.load_dataset`, its size, label set, and licence; fallback `stanfordnlp/sst2` (GLUE) —
      confirm availability + licence. Pick whichever is small, CPU-friendly, and clearly licensed.
- [ ] API to verify: current `transformers.Trainer` / `TrainingArguments` signature (arg names change
      across versions — e.g. `eval_strategy` vs `evaluation_strategy`); the recommended way to pass a
      `compute_metrics` fn; `AutoModelForSequenceClassification.from_pretrained(..., num_labels=…)`.
- [ ] Confirm `distilbert-base-uncased` is the right small checkpoint (size, license) or correct it.

## Acceptance criteria (each maps to evidence)
- [ ] AC1 (LO1–LO4) — chapter fine-tunes, evaluates vs. baseline, saves/reloads, and infers →
      evidence: the runnable script + training-curve + confusion-matrix artefacts.
- [ ] AC2 — every snippet runs → evidence: `check_snippets.py` pass + a real run log showing loss ↓
      and val-accuracy ↑ across epochs.
- [ ] AC3 — versions/dataset/API grounded → evidence: NOTE ids.
- [ ] AC4 — audience-fit: Java analogy for the training loop, every term explained, artefacts shown.
- [ ] AC5 — renders on GitHub → `check_markdown_render.py` pass; all `$…$`/```mermaid eyeballed.

## Gates
Entry: this spec approved; research NOTEs landed. Exit: all ACs satisfied; snippets run; links
resolve; fresh-Sonnet review sign-off; architect merge. (See `docs/definition-of-done.md`.)
