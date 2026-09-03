"""Semantic segmentation inference with a pretrained torchvision model — CPU only.

Companion script for:
  Machine Learning/Worked Examples/computer-vision/semantic-segmentation-coco.md

Environment (SPEC-ML-6 / NOTE-ML-5, shared ML virtualenv `.venv-ml`):
  torch==2.14.0+cpu
  torchvision==0.29.0+cpu
  matplotlib==3.11.1
  numpy==2.5.2

Run with the ML venv's interpreter, from anywhere (paths below are resolved relative to
this file, not to the current working directory):

    .venv-ml/Scripts/python.exe "Machine Learning/Worked Examples/computer-vision/code/segmentation_infer.py"

First run downloads:
  - the DeepLabV3 (ResNet-50 backbone) COCO-with-VOC-labels weights (~160 MB) via torchvision
    into the torch hub cache (`~/.cache/torch/hub/checkpoints/`, or
    `%USERPROFILE%\\.cache\\torch\\hub\\checkpoints\\` on Windows) — outside the repo, never
    committed;
  - one sample photo (~95 KB) from the official torchvision example gallery into
    ./datasets/_downloaded/segmentation/ next to this file (gitignored).

This script does NOT train anything — SPEC-ML-6 scopes training a segmentation model out of
this chapter entirely (conceptual only, contrast with ML-7's metrics). Every pixel of the mask
below comes from argmaxing a real forward pass through the pretrained model; nothing here is a
hand-authored or fabricated mask.
"""
from __future__ import annotations

import random
from pathlib import Path
import urllib.request

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

# --------------------------------------------------------------------------------------
# Reproducibility — same habit as mnist_cnn.py and detection_infer.py. A frozen, pretrained
# model in eval() mode is deterministic given a fixed input, but every RNG is still seeded
# so this script's behaviour never depends on process-level random state.
# --------------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths are resolved relative to this file so the script runs the same way regardless of
# the shell's current directory.
CHAPTER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CHAPTER_DIR / "code" / "datasets" / "_downloaded" / "segmentation"
ARTEFACTS_DIR = CHAPTER_DIR / "artefacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample image: the same photo ML-5's detection_infer.py runs its detector on — the exact file
# torchvision's own "Repurposing masks into bounding boxes" / visualization-utils example
# gallery uses (source: https://docs.pytorch.org/vision/stable/auto_examples/others/plot_visualization_utils.html,
# checked 2026-09-02 — that page loads it as `Path('../assets') / 'dog1.jpg'` from the same
# torchvision repo this raw URL points into; NOTE-ML-5 evidence #7 independently grounds this
# gallery directory as a COCO-style sample-image source). Reusing the identical file lets a
# reader compare ML-5's boxes and this chapter's per-pixel mask on the same picture.
SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
SAMPLE_IMAGE_PATH = DATA_DIR / "dog1.jpg"


def download_sample_image() -> Path:
    """Download the sample photo into DATA_DIR if not already cached; return its path."""
    if not SAMPLE_IMAGE_PATH.exists():
        print(f"Downloading {SAMPLE_IMAGE_URL} -> {SAMPLE_IMAGE_PATH}")
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_PATH)  # noqa: S310 - fixed, trusted URL
    return SAMPLE_IMAGE_PATH


def load_segmenter() -> tuple[torch.nn.Module, DeepLabV3_ResNet50_Weights, list[str]]:
    """Load the pretrained DeepLabV3 segmentation model and its class names.

    `weights.meta["categories"]` returns a 21-entry list (verified by printing it below):
    index 0 is "__background__", indices 1-20 are 20 Pascal VOC object classes. DEFAULT
    resolves to `COCO_WITH_VOC_LABELS_V1` — trained on the subset of COCO images that overlap
    VOC's 20 categories, re-labelled with VOC's category ids (NOTE-ML-5).
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    print(f"Weights: {weights}")
    print(f"Category list length: {len(categories)}")

    model = deeplabv3_resnet50(weights=weights, progress=True)
    model = model.to(DEVICE)
    model.eval()  # inference mode: disables dropout/batchnorm training behaviour
    return model, weights, categories


def run_inference(
    model: torch.nn.Module,
    weights: DeepLabV3_ResNet50_Weights,
    image_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the segmenter on one image; return (original uint8 image tensor, class-index mask)."""
    image_uint8 = decode_image(str(image_path))  # (3, H, W) uint8, RGB
    preprocess = weights.transforms()  # exact resize/normalise this model was trained with
    batch = preprocess(image_uint8).unsqueeze(0).to(DEVICE)  # (1, 3, 520, W')

    with torch.no_grad():
        logits = model(batch)["out"]  # (1, num_classes=21, H', W') raw, un-normalised scores

    # argmax over the class dimension: for every pixel, pick the class with the highest logit.
    # Because argmax is monotonic under softmax, this is identical to argmaxing probabilities —
    # no need to soften logits into a distribution just to find the winner (same reasoning as
    # mnist_cnn.py's evaluate()).
    mask = logits.argmax(dim=1).squeeze(0)  # (H', W') int64, values in [0, num_classes)
    return image_uint8, mask


def build_class_colors(num_classes: int) -> torch.Tensor:
    """A fixed, deterministic RGB colour per class index (tab20 colormap)."""
    cmap = plt.get_cmap("tab20", num_classes)
    colors = (torch.tensor([cmap(i)[:3] for i in range(num_classes)]) * 255).to(torch.uint8)
    return colors


def summarise_mask(mask: torch.Tensor, categories: list[str]) -> list[int]:
    """Print per-class pixel counts for every class actually present; return their indices."""
    present = torch.unique(mask).tolist()
    total_px = mask.numel()
    print(f"Classes present in mask ({total_px} px total):")
    for cls_idx in present:
        count = int((mask == cls_idx).sum())
        pct = 100.0 * count / total_px
        print(f"  {cls_idx:2d}: {categories[cls_idx]:<14s} {count:7d} px  ({pct:5.2f}%)")
    return present


def draw_overlay(
    image_uint8: torch.Tensor,
    mask: torch.Tensor,
    present_classes: list[int],
    class_colors: torch.Tensor,
    out_path: Path,
) -> torch.Tensor:
    """Alpha-blend the mask over the image and save it; return the resized image used underneath."""
    # draw_segmentation_masks needs a (num_masks, H, W) bool tensor — one plane per class to
    # draw — and an image at the *same* spatial size as the mask. The model's own preprocessing
    # resized the input to 520 on the shortest side, so resize the original image to match
    # before drawing, rather than resizing the mask back up (which would blur its hard edges).
    bool_masks = torch.stack([mask == cls_idx for cls_idx in present_classes])  # (K, H', W')
    colors = [tuple(class_colors[c].tolist()) for c in present_classes]

    image_resized = F.interpolate(
        image_uint8.unsqueeze(0).float(), size=tuple(mask.shape), mode="bilinear", align_corners=False
    ).squeeze(0).to(torch.uint8)

    overlay = draw_segmentation_masks(image_resized, masks=bool_masks, alpha=0.6, colors=colors)
    to_pil_image(overlay).save(out_path)
    return image_resized


def draw_legend(present_classes: list[int], categories: list[str], class_colors: torch.Tensor, out_path: Path) -> None:
    """Render a small swatch legend for exactly the classes detected in this mask."""
    fig, ax = plt.subplots(figsize=(4, 0.45 * len(present_classes) + 0.5))
    for i, cls_idx in enumerate(present_classes):
        color = tuple(c / 255 for c in class_colors[cls_idx].tolist())
        row = len(present_classes) - i - 1
        ax.add_patch(plt.Rectangle((0, row), 1, 1, color=color))
        ax.text(1.2, row + 0.5, categories[cls_idx], va="center", fontsize=11)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(present_classes))
    ax.axis("off")
    ax.set_title("Classes detected in mask", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Device: {DEVICE}")
    image_path = download_sample_image()
    model, weights, categories = load_segmenter()

    image_uint8, mask = run_inference(model, weights, image_path)
    print(f"Input image: {tuple(image_uint8.shape)} -> mask {tuple(mask.shape)}")

    present_classes = summarise_mask(mask, categories)
    class_colors = build_class_colors(len(categories))

    image_resized = draw_overlay(
        image_uint8, mask, present_classes, class_colors, ARTEFACTS_DIR / "segmentation_overlay.png"
    )
    print(f"Wrote {ARTEFACTS_DIR / 'segmentation_overlay.png'}")

    to_pil_image(image_resized).save(ARTEFACTS_DIR / "segmentation_original.png")
    print(f"Wrote {ARTEFACTS_DIR / 'segmentation_original.png'}")

    draw_legend(present_classes, categories, class_colors, ARTEFACTS_DIR / "segmentation_legend.png")
    print(f"Wrote {ARTEFACTS_DIR / 'segmentation_legend.png'}")


if __name__ == "__main__":
    main()
