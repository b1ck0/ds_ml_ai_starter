"""SPEC-ML-8: Text classification with a pretrained transformer encoder.

Environment (shared ML virtualenv, .venv-ml — verified 2026-09-02, research/NOTE-ML-7-nlp-models.md):
    torch==2.14.0+cpu
    transformers==5.16.1
    scikit-learn==1.9.0
    matplotlib==3.11.1
    Python 3.13.7, CPU only.

Model: distilbert/distilbert-base-uncased-finetuned-sst-2-english (Apache-2.0), already fine-tuned
for binary sentiment classification on SST-2. First run downloads ~268MB to the local HuggingFace
cache (~/.cache/huggingface/hub); subsequent runs load from cache.

Run:
    .venv-ml/Scripts/python.exe "Machine Learning/Worked Examples/natural-language/code/text_classification.py"
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — this script only writes PNGs, never shows a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
HERE = Path(__file__).resolve().parent
ARTEFACTS = HERE.parent / "artefacts"
ARTEFACTS.mkdir(parents=True, exist_ok=True)

# Inference on this model is deterministic (no sampling), but we seed anyway — the moment a reader
# swaps in a generative model or adds dropout-based augmentation, reproducibility stops being free.
torch.manual_seed(42)
np.random.seed(42)


def section_pipeline_api() -> None:
    """LO2 — the fast path: `pipeline()` hides tokenizer + model + softmax behind one call."""
    print("\n=== 1. pipeline() API ===")
    classifier = pipeline("text-classification", model=MODEL_ID, device="cpu")
    samples = [
        "This movie was a masterpiece from start to finish.",
        "I want my two hours back. Absolutely dreadful.",
    ]
    results = classifier(samples)
    for text, result in zip(samples, results):
        print(f"{result['label']:8s} score={result['score']:.4f}  {text!r}")


def section_automodel_api(tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification) -> None:
    """LO2 — the explicit path: tokenizer -> model -> logits -> softmax -> label.

    This is what `pipeline()` does internally; reading it once makes the pipeline's output legible
    instead of magic, the way reading a JDBC `ResultSet` loop once demystifies an ORM.
    """
    print("\n=== 2. AutoTokenizer + AutoModelForSequenceClassification (explicit) ===")
    text = "The plot dragged, but the acting saved it."
    encoded = tokenizer(text, return_tensors="pt")
    print("input_ids shape:", tuple(encoded["input_ids"].shape))
    print("tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

    with torch.no_grad():
        logits = model(**encoded).logits
    print("raw logits:", logits.numpy().round(4))

    probs = torch.softmax(logits, dim=-1)[0]
    print("softmax probs:", probs.numpy().round(4))
    print("id2label:", model.config.id2label)

    pred_id = int(torch.argmax(probs))
    print(f"predicted: {model.config.id2label[pred_id]} (confidence={probs[pred_id]:.4f})")


# --- LO3: a small hand-written labelled evaluation set (per NOTE-ML-7, 15-30 hand-written examples). ---
# label: 0 = NEGATIVE, 1 = POSITIVE — matches this model's id2label (confirmed by section_automodel_api
# above; SST-2 is movie-review sentiment, so examples lean toward that register). Deliberately mixes
# easy cases with harder ones (negation, mixed sentiment, understatement) to produce real errors,
# not just a clean 100% score.
EVAL_SET: list[tuple[str, int]] = [
    ("This movie was a masterpiece from start to finish.", 1),
    ("I want my two hours back. Absolutely dreadful.", 0),
    ("The plot dragged, but the acting saved it.", 1),
    ("One of the worst films I have ever sat through.", 0),
    ("A charming, witty, endlessly rewatchable comedy.", 1),
    ("The dialogue was wooden and the pacing was glacial.", 0),
    ("I've never laughed so hard in a theater.", 1),
    ("Not the worst thing I've seen, but close to it.", 0),
    ("A visually stunning film with a story to match.", 1),
    ("Predictable, overlong, and badly edited.", 0),
    ("This is not a bad movie at all — quite enjoyable.", 1),
    ("It was fine, nothing special, forgettable really.", 0),
    ("An absolute triumph of storytelling and performance.", 1),
    ("Two hours of my life I will never get back.", 0),
    ("The cinematography alone is worth the ticket price.", 1),
    ("A confusing mess that never finds its footing.", 0),
    ("Surprisingly good given the low expectations going in.", 1),
    ("I struggled to stay awake through the second act.", 0),
    ("Every performance in this cast is pitch perfect.", 1),
    ("The sequel fails to capture any of the original's magic.", 0),
    ("A small gem that deserves a much wider audience.", 1),
    ("Loud, dumb, and proud of it — skip this one.", 0),
    ("Genuinely moving without ever feeling manipulative.", 1),
    ("The special effects can't disguise the lazy script.", 0),
]


def section_evaluate(tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification) -> None:
    """LO3 — batch inference on the eval set + DS-6 metrics (accuracy, precision, recall, F1,
    confusion matrix) + write the predictions table and confusion matrix artefacts."""
    print("\n=== 3. Evaluate on the hand-written labelled set ===")
    texts = [t for t, _ in EVAL_SET]
    y_true = [label for _, label in EVAL_SET]

    start = time.perf_counter()
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    elapsed = time.perf_counter() - start

    probs = torch.softmax(logits, dim=-1)
    y_pred = torch.argmax(probs, dim=-1).tolist()
    confidences = probs.max(dim=-1).values.tolist()

    print(f"batch inference on {len(texts)} examples took {elapsed:.3f}s on CPU "
          f"({elapsed / len(texts) * 1000:.1f}ms/example)")

    # --- predictions table artefact ---
    id2label = model.config.id2label
    pred_rows = []
    for text, true_id, pred_id, conf in zip(texts, y_true, y_pred, confidences):
        pred_rows.append({
            "text": text,
            "true_label": id2label[true_id],
            "pred_label": id2label[pred_id],
            "confidence": round(conf, 4),
            "correct": true_id == pred_id,
        })

    predictions_csv = ARTEFACTS / "predictions.csv"
    with open(predictions_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "true_label", "pred_label", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(pred_rows)
    print(f"wrote {predictions_csv}")

    # --- DS-6 metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall:    {rec:.4f}")
    print(f"f1:        {f1:.4f}")
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["NEGATIVE", "POSITIVE"]))

    metrics_csv = ARTEFACTS / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", round(acc, 4)])
        writer.writerow(["precision", round(prec, 4)])
        writer.writerow(["recall", round(rec, 4)])
        writer.writerow(["f1", round(f1, 4)])
        writer.writerow(["n_examples", len(texts)])
    print(f"wrote {metrics_csv}")

    # --- confusion matrix artefact ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NEGATIVE", "POSITIVE"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"DistilBERT-SST2 on {len(texts)} hand-written examples\naccuracy={acc:.3f}  f1={f1:.3f}")
    fig.tight_layout()
    cm_path = ARTEFACTS / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"wrote {cm_path}")

    # Print the misclassified rows so the chapter can quote real errors, not fabricated ones.
    print("\nmisclassified examples:")
    for row in pred_rows:
        if not row["correct"]:
            print(f"  true={row['true_label']:8s} pred={row['pred_label']:8s} "
                  f"conf={row['confidence']:.3f}  {row['text']!r}")


def main() -> None:
    print(f"Loading tokenizer + model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()  # inference mode: disables dropout — this model has none active at inference by
                  # default, but it's the correct habit for every model, including ones that do.

    section_pipeline_api()
    section_automodel_api(tokenizer, model)
    section_evaluate(tokenizer, model)


if __name__ == "__main__":
    main()
