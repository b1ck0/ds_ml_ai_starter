"""MNIST digit classification with a small CNN — PyTorch + torchvision, CPU only.

Companion script for:
  Machine Learning/Worked Examples/computer-vision/image-classification-mnist.md

Environment (SPEC-ML-4 / NOTE-ML-1, shared ML virtualenv `.venv-ml`):
  torch==2.14.0+cpu
  torchvision==0.29.0+cpu
  matplotlib==3.11.1
  scikit-learn==1.9.0
  numpy==2.5.2

Run with the ML venv's interpreter, from anywhere (paths below are resolved relative to
this file, not to the current working directory):

    .venv-ml/Scripts/python.exe "Machine Learning/Worked Examples/computer-vision/code/mnist_cnn.py"

First run downloads MNIST (~11 MB) via torchvision into ./datasets/_downloaded/mnist next to this
file (gitignored — see NOTE-ML-1 on mirror reliability). Trains for 3 epochs on CPU (a few minutes),
then writes three artefacts (training curve, confusion matrix, sample-prediction grid) plus the
trained weights into ../artefacts/.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# --------------------------------------------------------------------------------------
# Reproducibility — same idea as seeding java.util.Random in a test: fix every source of
# randomness (Python's random module, NumPy, and PyTorch's own RNG) so the numbers printed
# in the chapter's prose are the numbers you get too.
# --------------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths are resolved relative to this file so the script runs the same way regardless of
# the shell's current directory.
CHAPTER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CHAPTER_DIR / "code" / "datasets" / "_downloaded" / "mnist"
ARTEFACTS_DIR = CHAPTER_DIR / "artefacts"
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# --------------------------------------------------------------------------------------
# 1. Data — torchvision datasets + the v2 transforms API (NOTE-ML-1: v2 is the recommended,
#    forward-compatible API as of torchvision 0.29.0).
# --------------------------------------------------------------------------------------
transform = v2.Compose([
    v2.ToImage(),                          # PIL image -> tv_tensors.Image (uint8, CxHxW)
    v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float32 [0.0, 1.0]
])

train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# --------------------------------------------------------------------------------------
# 2. Model — a small CNN as an nn.Module: conv -> relu -> pool, twice, then two fully
#    connected layers down to 10 class logits (one per digit).
# --------------------------------------------------------------------------------------
class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Block 1: 1x28x28 -> 16x28x28 (conv, padding=1 keeps spatial size) -> 16x14x14 (pool)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Block 2: 16x14x14 -> 32x14x14 (conv) -> 32x7x7 (pool)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # halves H and W each time it's applied
        self.relu = nn.ReLU()
        # Flattened conv output: 32 channels * 7 * 7 spatial positions
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 digit classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))  # (B, 1, 28, 28)  -> (B, 16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 16, 14, 14) -> (B, 32, 7, 7)
        x = torch.flatten(x, start_dim=1)        # (B, 32, 7, 7)   -> (B, 1568)
        x = self.relu(self.fc1(x))               # (B, 1568)       -> (B, 128)
        return self.fc2(x)                       # (B, 128)        -> (B, 10) raw logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# 3. Train — the loop: forward pass, CrossEntropy loss, backward pass, optimizer step.
# --------------------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float]:
    """Return (mean loss, accuracy) over `loader`, in eval mode with gradients off."""
    model.eval()  # turns off dropout/batchnorm training behaviour (this model has neither,
    # but the habit matters: forgetting eval() is Pitfall #1 in the chapter)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # no autograd graph needed for evaluation -> saves memory/time
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train() -> dict:
    model = MnistCNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {DEVICE}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Train batches/epoch: {len(train_loader)} (batch size {BATCH_SIZE})")

    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()  # dropout/batchnorm would switch to training behaviour here
        running_loss = 0.0
        running_count = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()          # clear gradients from the previous step —
            # PyTorch accumulates grads by default; skipping this adds each batch's
            # gradient on top of the last, which silently corrupts training.
            logits = model(images)         # forward pass: raw, unnormalized class scores
            loss = loss_fn(logits, labels)  # CrossEntropyLoss applies log-softmax + NLL
            loss.backward()                # backward pass: populates .grad on every parameter
            optimizer.step()               # apply one gradient-descent update to the weights

            running_loss += loss.item() * images.size(0)
            running_count += images.size(0)

            if batch_idx % 200 == 0:
                print(f"  epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                      f"running_loss={running_loss / running_count:.4f}")

        train_loss = running_loss / running_count
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        print(f"epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  "
              f"test_loss={test_loss:.4f}  test_accuracy={test_acc:.4f}")

    elapsed = time.perf_counter() - start
    print(f"Training wall-clock: {elapsed:.1f}s on {DEVICE}")

    torch.save(model.state_dict(), ARTEFACTS_DIR / "mnist_cnn.pt")
    return {"model": model, "history": history, "elapsed": elapsed}


# --------------------------------------------------------------------------------------
# 4. Evaluate — final test accuracy + confusion matrix.
# --------------------------------------------------------------------------------------
def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (images as a single stacked array, true labels, predicted labels) for `loader`."""
    model.eval()
    all_images, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            preds = logits.argmax(dim=1).cpu()
            all_images.append(images)
            all_labels.append(labels)
            all_preds.append(preds)
    return (
        torch.cat(all_images).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
    )


def plot_training_curve(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train loss")
    ax_loss.plot(epochs, history["test_loss"], marker="o", label="test loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("CrossEntropy loss")
    ax_loss.set_title("Loss per epoch")
    ax_loss.set_xticks(list(epochs))
    ax_loss.legend()

    ax_acc.plot(epochs, [a * 100 for a in history["test_accuracy"]], marker="o", color="green")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_title("Test accuracy per epoch")
    ax_acc.set_xticks(list(epochs))
    ax_acc.set_ylim(90, 100)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    ax.set_title("MNIST test-set confusion matrix (10,000 images)")
    for i in range(10):
        for j in range(10):
            value = cm[i, j]
            colour = "white" if value > cm.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=colour, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_sample_predictions(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                             path: Path, n: int = 16, n_errors: int = 4, seed: int = SEED) -> None:
    """Grid of `n` sample predictions. Deliberately forces up to `n_errors` misclassified
    digits into the grid (instead of pure random sampling) — with ~98.9% accuracy, a plain
    random sample of 16 images is often all-correct and teaches nothing about failure modes.
    """
    rng = np.random.default_rng(seed)
    wrong_idx = np.flatnonzero(y_true != y_pred)
    n_errors = min(n_errors, len(wrong_idx))
    error_idx = rng.choice(wrong_idx, size=n_errors, replace=False) if n_errors else np.array([], dtype=int)

    correct_idx_pool = np.flatnonzero(y_true == y_pred)
    correct_idx = rng.choice(correct_idx_pool, size=n - n_errors, replace=False)

    idx = np.concatenate([correct_idx, error_idx])
    rng.shuffle(idx)  # interleave errors among the correct predictions instead of grouping them
    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for ax, i in zip(axes.flat, idx):
        img = images[i].squeeze(0)  # (1, 28, 28) -> (28, 28)
        ax.imshow(img, cmap="gray")
        correct = y_true[i] == y_pred[i]
        colour = "green" if correct else "red"
        ax.set_title(f"true={y_true[i]} pred={y_pred[i]}", color=colour, fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample test predictions (green = correct, red = wrong)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = train()
    model, history, elapsed = result["model"], result["history"], result["elapsed"]

    final_test_loss, final_test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
    print(f"\nFinal test accuracy: {final_test_acc:.4f}  (test loss {final_test_loss:.4f})")

    images, y_true, y_pred = collect_predictions(model, test_loader)

    plot_training_curve(history, ARTEFACTS_DIR / "mnist_training_curve.png")
    cm = plot_confusion_matrix(y_true, y_pred, ARTEFACTS_DIR / "mnist_confusion_matrix.png")
    plot_sample_predictions(images, y_true, y_pred, ARTEFACTS_DIR / "mnist_sample_predictions.png")

    misclassified = int((y_true != y_pred).sum())
    print(f"Misclassified: {misclassified} / {len(y_true)}")
    print("Per-class confusion matrix diagonal (correct counts):", np.diag(cm).tolist())
    print(f"Wall-clock training time: {elapsed:.1f}s ({EPOCHS} epochs, device={DEVICE})")
    print(f"Artefacts written to: {ARTEFACTS_DIR}")


if __name__ == "__main__":
    main()
