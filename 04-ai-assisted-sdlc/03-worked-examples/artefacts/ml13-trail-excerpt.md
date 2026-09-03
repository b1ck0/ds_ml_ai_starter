# Artefact — SPEC-ML-13's real spec → NOTE → commit trail

*Referenced from [`02-how-this-repo-was-built.md`](../02-how-this-repo-was-built.md) §2. Every
excerpt below is copied verbatim from a real, committed file at HEAD or a real command run against
this repository — nothing here is reconstructed or paraphrased for effect. Reproduce any piece with
the command shown above it.*

## 1. The spec's "Claims to ground" (must land as NOTEs before writing starts)

```text
$ sed -n '70,80p' specs/SPEC-ML-13-fine-tuning-a-transformer.md
```

```markdown
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
```

## 2. The grounding NOTE's architect-correction banner

```text
$ sed -n '1,11p' research/NOTE-ML-13-transformers-api-and-versions.md
```

```markdown
# NOTE-ML-13: Transformers fine-tuning — package versions, datasets, API signatures

> **ARCHITECT CORRECTION (2026-09-03):** the original draft of this note misread PyPI and pinned
> `transformers` **4.41.2**, which is a stale major version. The real current stable is
> **`transformers==5.16.1`** (verified against https://pypi.org/pypi/transformers/json, 2026-09-03) —
> the same version grounded for ML-14, so the two adjacent chapters stay consistent. Every "4.41.2"
> below should read **5.16.1**. The Trainer / `eval_strategy` / `compute_metrics`-to-`Trainer()` /
> `AutoModelForSequenceClassification(..., num_labels=…)` guidance is still correct in 5.x, but the
> writer MUST confirm it against the actually-installed 5.16.1 by running the code, and fix any drift
> from the 5.0 major bump. `torch==2.14.0` and `datasets==5.0.1` are unchanged.
```

## 3. The grounding notes landing (four notes, one Haiku dispatch)

```text
$ git log -1 273c8e4 --format="%h %ad %s" --date=format:"%Y-%m-%d %H:%M"
273c8e4 2026-09-03 15:02 Add grounding notes for ML-13, ML-14, ML-15, DS-19 (Haiku research)
```

## 4. The chapter commit, untrimmed

```text
$ git show --stat 90b1ea0
```

```text
commit 90b1ea0926988a7127faaa325e9c552d6c7db5df
Author: Vasil Yordanov <vasil.yordanov88@gmail.com>
Date:   Thu Sep 3 16:12:47 2026 +0300

    Add ML-13: Fine-tuning a transformer end to end (new worked-example chapter)

    Closes the gap that the NLP chapters were inference-only: fine-tunes DistilBERT
    on dair-ai/emotion (6-way) with the explicit PyTorch loop shown once, then the HF
    Trainer. Real measured run (synced into the prose): untrained-head baseline
    0.0335 acc / 0.0127 macro-F1 -> fine-tuned 0.9235 acc / 0.8824 macro-F1 over 3
    epochs (train-slice 0.98 shows an honest overfit gap); a learning-rate pitfall
    demo where a sane 5e-5 (0.7733) clearly beats a too-high 1e-2 (0.3517); an honest
    low-confidence miss kept as a teaching case. Trains on CPU in ~15 min; saved model
    is gitignored (regenerate by rerunning).

    Pinned transformers==5.16.1 (real current; the ML-13 note's 4.41.2 was an
    architect-corrected misread) + datasets==5.0.1 -- documents a real 5.x drift where
    whole-column access returns a Column, not a list. ML-8 now points forward here
    instead of claiming fine-tuning needs a GPU. Its own confusion-matrix artefact is
    namespaced (finetune_confusion_matrix.png) so it doesn't clobber ML-8's. Both gates
    pass; curriculum + ML README already carried the entry from the ML-15 commit.

    Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

 .gitignore                                                                        |   4 +
 02-machine-learning/03-worked-examples/02-natural-language/01-text-classification.md  |  17 +-
 02-machine-learning/03-worked-examples/02-natural-language/03-fine-tuning-a-transformer.md | 591 ++++
 02-machine-learning/03-worked-examples/02-natural-language/artefacts/finetune_confusion_matrix.png | Bin 0 -> 56819 bytes
 02-machine-learning/03-worked-examples/02-natural-language/artefacts/metrics_summary.csv | 6 +
 02-machine-learning/03-worked-examples/02-natural-language/artefacts/new_sentence_predictions.csv | 7 +
 02-machine-learning/03-worked-examples/02-natural-language/artefacts/training_curve.png | Bin 0 -> 71565 bytes
 02-machine-learning/03-worked-examples/02-natural-language/code/fine_tune_classifier.py | 448 +++
 8 files changed, 1066 insertions(+), 7 deletions(-)
```

## 5. The chapter's own "Environment note for the architect" (the writer's honest hand-off)

```text
$ sed -n '544,557p' "02-machine-learning/03-worked-examples/02-natural-language/03-fine-tuning-a-transformer.md"
```

```markdown
### Environment note (for the architect)

No discrepancies against the NOTE-ML-13 architect-corrected banner: `transformers==5.16.1` (not the
original draft's stale 4.41.2), `torch==2.14.0+cpu`, `datasets==5.0.1`, `evaluate==0.4.6`,
`scikit-learn==1.9.0`, `accelerate==1.14.0` — all confirmed already installed in the shared `.venv-ml`
and used as-is; `eval_strategy`, `compute_metrics`-to-`Trainer()`, and
`AutoModelForSequenceClassification(..., num_labels=…)` all behaved exactly as the NOTE's guidance
said, confirmed by running them, not assumed.

One genuine 5.x-era API drift was found and fixed, documented inline in Section 3.2: `datasets==5.0.1`
returns a `datasets.arrow_dataset.Column` (not a `list`) from whole-column access, which
`transformers==5.16.1`'s tokenizer rejects; every call site in `code/fine_tune_classifier.py` now
wraps a full column in `list(...)` before it reaches the tokenizer.
```

This is the same trail Section 2 walks in prose; this artefact exists so the excerpts can be diffed
byte-for-byte against the live files rather than trusted on the chapter's word.
