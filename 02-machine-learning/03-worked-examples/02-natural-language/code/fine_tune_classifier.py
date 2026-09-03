"""SPEC-ML-13: Fine-tuning a transformer — training a real model end to end.

Environment (shared ML virtualenv, .venv-ml; verified 2026-09-03,
research/NOTE-ML-13-transformers-api-and-versions.md, corrected by the architect banner at the
top of that NOTE):
    torch==2.14.0+cpu
    transformers==5.16.1
    datasets==5.0.1
    evaluate==0.4.6
    scikit-learn==1.9.0
    accelerate==1.14.0
    matplotlib==3.11.1
    Python 3.13.7, CPU only.

Dataset: dair-ai/emotion (HF Hub), split version — 16,000 train / 2,000 validation / 2,000 test,
6 emotion labels (sadness, joy, love, anger, fear, surprise). Licence on the HF Hub is listed as
"other" — "for educational and research purposes only"; fine for this teaching example, substitute
a properly licensed dataset for anything production-bound.

Model: distilbert-base-uncased (Apache-2.0), a BASE encoder with no classification head — this
script attaches a fresh, untrained head (`num_labels=6`) and trains it (and the encoder body)
against dair-ai/emotion.

Run:
    .venv-ml/Scripts/python.exe \
        "02-machine-learning/03-worked-examples/02-natural-language/code/fine_tune_classifier.py"

This is a real, full run: several minutes on a modern CPU (measured wall-clock is printed at the
end and quoted in the chapter — not estimated). No GPU is used or required.
"""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this script only writes PNGs, never shows a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import optim
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# --- constants -------------------------------------------------------------------------------
SEED = 42
MODEL_ID = "distilbert-base-uncased"
DATASET_ID = "dair-ai/emotion"
MAX_LENGTH = 64

N_TRAIN = 5_000            # subset of the 16,000-row train split — NOTE-ML-13's "5-10K" floor
N_EVAL_DURING_TRAINING = 600  # subset of the validation split, used for per-epoch checkpoints
EPOCHS = 3
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

N_EXPLICIT_DEMO = 200       # tiny slice for the hand-written PyTorch loop (Section 4)
N_BAD_LR_DEMO = 1_000       # slice for the "learning rate too high" pitfall demo — needs enough
                            # steps for the *normal*-LR run to visibly separate from a majority-
                            # class collapse, or the demo fails to make its point (see chapter note)
BAD_LR_DEMO_EPOCHS = 2
BAD_LR_DEMO_BATCH_SIZE = 25
N_TRAIN_EVAL_SLICE = 400    # slice of the *training* data re-scored to expose train/test gap

HERE = Path(__file__).resolve().parent
ARTEFACTS = HERE.parent / "artefacts"
ARTEFACTS.mkdir(parents=True, exist_ok=True)
SAVE_DIR = HERE / "fine_tuned_model"  # gitignored (see .gitignore) — regenerate by rerunning this file


def set_seed(seed: int) -> None:
    """Seed every source of randomness this script touches, mirroring SPEC-ML-4's convention."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- data --------------------------------------------------------------------------------------

def load_data():
    """Load dair-ai/emotion, print real sizes/label distribution, return raw + tokenized splits."""
    print("\n=== 1. Load dair-ai/emotion ===")
    ds = load_dataset(DATASET_ID)
    label_names = ds["train"].features["label"].names
    print(f"splits: { {k: len(v) for k, v in ds.items()} }")
    print(f"labels: {label_names}")

    counts = {name: 0 for name in label_names}
    for lab in ds["train"]["label"]:
        counts[label_names[lab]] += 1
    print(f"train label distribution (of {len(ds['train'])}): {counts}")

    example = ds["train"][0]
    print(f"example row: {example}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    encoded_example = tokenizer(example["text"], return_tensors="pt")
    print("tokens for that row:", tokenizer.convert_ids_to_tokens(encoded_example["input_ids"][0]))

    train_raw = ds["train"].shuffle(seed=SEED).select(range(N_TRAIN))
    eval_raw = ds["validation"].shuffle(seed=SEED).select(range(N_EVAL_DURING_TRAINING))
    test_raw = ds["test"]  # full 2,000-row held-out test split — used only for final comparison

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_tok = train_raw.map(tokenize, batched=True)
    eval_tok = eval_raw.map(tokenize, batched=True)

    return tokenizer, label_names, train_raw, eval_raw, test_raw, train_tok, eval_tok


# --- Section 4a: the explicit PyTorch loop, run once, on a tiny slice --------------------------

def show_explicit_loop(tokenizer, label_names, train_raw) -> None:
    """LO3 (part 1) — forward / loss / backward / step / zero_grad, spelled out by hand.

    A throwaway model instance, trained for exactly one epoch over N_EXPLICIT_DEMO examples, so
    the reader watches every line `Trainer` will hide in Section 4b. This model is discarded —
    it never touches the real fine-tuning run below.
    """
    print("\n=== 2. The explicit PyTorch loop (demo only, discarded afterwards) ===")
    demo_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=len(label_names))
    demo_model.train()

    texts = train_raw["text"][:N_EXPLICIT_DEMO]
    labels = train_raw["label"][:N_EXPLICIT_DEMO]
    encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    labels_t = torch.tensor(labels)

    optimizer = optim.AdamW(demo_model.parameters(), lr=5e-5)
    batch_size = 20
    n_steps = len(texts) // batch_size

    for step in range(n_steps):
        s, e = step * batch_size, (step + 1) * batch_size
        batch_inputs = {k: v[s:e] for k, v in encoded.items()}
        batch_labels = labels_t[s:e]

        optimizer.zero_grad()                                    # clear last step's gradients
        outputs = demo_model(**batch_inputs, labels=batch_labels)  # forward pass + loss in one call
        loss = outputs.loss                                       # CrossEntropyLoss, computed inside the model
        loss.backward()                                           # autograd: assign blame to every weight
        optimizer.step()                                          # nudge every weight downhill

        print(f"  step {step + 1:2d}/{n_steps}  loss={loss.item():.4f}")

    del demo_model


# --- Section 4b: the "too high a learning rate" pitfall, demonstrated cheaply ------------------

def demo_bad_learning_rate(tokenizer, label_names, train_raw, eval_raw) -> dict:
    """LO5 — train two tiny models, one at the default LR and one 200x too high; compare eval acc."""
    print("\n=== 3. Pitfall demo: learning rate too high ===")
    texts = train_raw["text"][:N_BAD_LR_DEMO]
    labels = train_raw["label"][:N_BAD_LR_DEMO]
    encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    labels_t = torch.tensor(labels)

    # NOTE: datasets==5.0.1 returns a `datasets.arrow_dataset.Column` object (not a plain `list`)
    # from `dataset["col"]` when the WHOLE column is read — transformers==5.16.1's tokenizer
    # rejects that type outright (`list(...)` fixes it; a *slice* of a Column, like `col[:10]`,
    # already comes back as a plain `list`, which is why the slices above never needed this).
    eval_texts = list(eval_raw["text"])
    eval_labels = list(eval_raw["label"])
    eval_enc = tokenizer(eval_texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")

    results = {}
    for tag, lr in [("normal (5e-5)", 5e-5), ("too high (1e-2)", 1e-2)]:
        m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=len(label_names))
        m.train()
        opt = optim.AdamW(m.parameters(), lr=lr)
        for epoch in range(BAD_LR_DEMO_EPOCHS):
            for step in range(len(texts) // BAD_LR_DEMO_BATCH_SIZE):
                s, e = step * BAD_LR_DEMO_BATCH_SIZE, (step + 1) * BAD_LR_DEMO_BATCH_SIZE
                opt.zero_grad()
                out = m(
                    input_ids=encoded["input_ids"][s:e],
                    attention_mask=encoded["attention_mask"][s:e],
                    labels=labels_t[s:e],
                )
                out.loss.backward()
                opt.step()

        m.eval()
        with torch.no_grad():
            logits = m(**eval_enc).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        acc = accuracy_score(eval_labels, preds)
        final_loss = out.loss.item()
        print(f"  {tag:18s} final_train_loss={final_loss:.4f}  eval_accuracy={acc:.4f}")
        results[tag] = {"final_train_loss": final_loss, "eval_accuracy": acc}
        del m

    return results


# --- Section 5: the real fine-tuning run, via Trainer -------------------------------------------

def build_trainer(model, train_tok, eval_tok, tokenizer):
    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(HERE / "_trainer_scratch"),  # ephemeral logs/checkpoints — see .gitignore
        eval_strategy="epoch",       # transformers 5.x: NOT the deprecated `evaluation_strategy`
        logging_strategy="epoch",    # one training-loss row per epoch, matching eval_strategy
        save_strategy="no",          # this chapter saves explicitly (Section 6) — no mid-run checkpoints
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        seed=SEED,
        report_to=[],
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        compute_metrics=compute_metrics,  # passed to Trainer, NOT TrainingArguments (5.x, confirmed)
        data_collator=collator,
    )


def plot_training_curve(log_history: list[dict]) -> None:
    epochs, train_loss, eval_acc = [], [], []
    for row in log_history:
        if "loss" in row and "epoch" in row and "eval_loss" not in row:
            epochs.append(row["epoch"])
            train_loss.append(row["loss"])
        if "eval_accuracy" in row:
            eval_acc.append(row["eval_accuracy"])

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(epochs, train_loss, "o-", color="tab:red", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(epochs[: len(eval_acc)], eval_acc, "s-", color="tab:blue", label="validation accuracy")
    ax2.set_ylabel("validation accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle(
        f"Fine-tuning distilbert-base-uncased on dair-ai/emotion\n({N_TRAIN} examples, {EPOCHS} epochs)"
    )
    fig.tight_layout()
    out = ARTEFACTS / "training_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# --- Section 6: evaluate, compare to baseline, save/reload, infer ------------------------------

def evaluate_texts(model, tokenizer, texts, y_true, batch_size=128):
    """Batched inference over a list of raw texts; returns (y_pred, confidences)."""
    model.eval()
    y_pred: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encoded = tokenizer(chunk, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            y_pred.extend(preds.tolist())
            confidences.extend(probs.max(dim=-1).values.tolist())
    return y_pred, confidences


def score_and_report(model, tokenizer, texts, y_true, label_names, title) -> dict:
    y_pred, _ = evaluate_texts(model, tokenizer, texts, y_true)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n--- {title} ---")
    print(f"accuracy:  {acc:.4f}")
    print(f"macro-F1:  {f1_macro:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    return {"accuracy": acc, "f1_macro": f1_macro, "y_pred": y_pred}


def plot_confusion(y_true, y_pred, label_names, title, filename) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    out = ARTEFACTS / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def run_inference_demo(model, tokenizer, label_names) -> None:
    print("\n=== 8. Inference on brand-new sentences (reloaded model) ===")
    samples = [
        "I can't believe I actually won the competition, I'm over the moon!",
        "I keep checking the door, certain someone is about to break in.",
        "Losing my grandmother's ring in the move has left me hollow.",
        "He slammed the laptop shut and stormed out of the meeting.",
        "Out of nowhere, the whole office started singing happy birthday to me.",
        "I love how the rain sounds against the window at night.",
    ]
    encoded = tokenizer(samples, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).tolist()
    rows = []
    for text, pid, p in zip(samples, pred_ids, probs):
        conf = p[pid].item()
        print(f"  {label_names[pid]:8s} conf={conf:.4f}  {text!r}")
        rows.append({"text": text, "predicted_label": label_names[pid], "confidence": round(conf, 4)})

    out = ARTEFACTS / "new_sentence_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


def main() -> None:
    overall_start = time.perf_counter()
    set_seed(SEED)

    tokenizer, label_names, train_raw, eval_raw, test_raw, train_tok, eval_tok = load_data()

    # test_raw is the FULL, un-sliced test split, so `test_raw["text"]`/`["label"]` come back as
    # `datasets.arrow_dataset.Column` objects (datasets==5.0.1) rather than plain lists — cast
    # once here to plain lists, the type transformers==5.16.1's tokenizer actually accepts.
    test_texts = list(test_raw["text"])
    test_labels = list(test_raw["label"])

    show_explicit_loop(tokenizer, label_names, train_raw)
    bad_lr_results = demo_bad_learning_rate(tokenizer, label_names, train_raw, eval_raw)

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    # --- baseline: same architecture, freshly-initialized head, ZERO fine-tuning steps ---
    print("\n=== 4. Baseline: pretrained body + untrained head, evaluated as-is ===")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(label_names), id2label=id2label, label2id=label2id,
    )
    baseline_metrics = score_and_report(
        baseline_model, tokenizer, test_texts, test_labels, label_names,
        title="BASELINE (no fine-tuning) on the 2,000-row test set",
    )
    del baseline_model

    # --- the real fine-tuning run ---
    print("\n=== 5. Fine-tune (Trainer) ===")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(label_names), id2label=id2label, label2id=label2id,
    )
    trainer = build_trainer(model, train_tok, eval_tok, tokenizer)
    train_start = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_start
    print(f"training wall-clock: {train_elapsed:.1f}s for {N_TRAIN} examples x {EPOCHS} epochs")
    plot_training_curve(trainer.state.log_history)

    # --- evaluate the fine-tuned model on the held-out test set ---
    print("\n=== 6. Evaluate the fine-tuned model on the held-out test set ===")
    finetuned_metrics = score_and_report(
        model, tokenizer, test_texts, test_labels, label_names,
        title="FINE-TUNED on the 2,000-row test set",
    )
    plot_confusion(
        test_labels, finetuned_metrics["y_pred"], label_names,
        title=f"Fine-tuned DistilBERT on dair-ai/emotion test set\n"
              f"accuracy={finetuned_metrics['accuracy']:.3f}  macro-F1={finetuned_metrics['f1_macro']:.3f}",
        # NOTE: this chapter shares `artefacts/` with SPEC-ML-8's text-classification chapter,
        # which already owns the filename `confusion_matrix.png` — a distinct name here avoids
        # silently overwriting that chapter's artefact on every rerun.
        filename="finetune_confusion_matrix.png",
    )

    # --- pitfall: evaluating on train inflates the number ---
    print("\n=== 7. Pitfall demo: train-set accuracy vs held-out test accuracy ===")
    train_slice_texts = train_raw["text"][:N_TRAIN_EVAL_SLICE]
    train_slice_labels = train_raw["label"][:N_TRAIN_EVAL_SLICE]
    train_eval_metrics = score_and_report(
        model, tokenizer, train_slice_texts, train_slice_labels, label_names,
        title=f"FINE-TUNED on {N_TRAIN_EVAL_SLICE} rows it TRAINED ON",
    )

    # --- save + reload + infer ---
    print(f"\nSaving model + tokenizer to {SAVE_DIR}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"Reloading from {SAVE_DIR}")
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    reloaded_model.eval()
    run_inference_demo(reloaded_model, reloaded_tokenizer, label_names)

    # --- write a summary comparison table ---
    summary_csv = ARTEFACTS / "metrics_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["evaluation", "accuracy", "f1_macro"])
        writer.writerow(["baseline_untrained_head", round(baseline_metrics["accuracy"], 4),
                          round(baseline_metrics["f1_macro"], 4)])
        writer.writerow(["fine_tuned_test_set", round(finetuned_metrics["accuracy"], 4),
                          round(finetuned_metrics["f1_macro"], 4)])
        writer.writerow(["fine_tuned_on_train_slice", round(train_eval_metrics["accuracy"], 4),
                          round(train_eval_metrics["f1_macro"], 4)])
        for tag, r in bad_lr_results.items():
            writer.writerow([f"bad_lr_demo[{tag}]", round(r["eval_accuracy"], 4), ""])
    print(f"wrote {summary_csv}")

    overall_elapsed = time.perf_counter() - overall_start
    print(f"\nTOTAL wall-clock for this script: {overall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
