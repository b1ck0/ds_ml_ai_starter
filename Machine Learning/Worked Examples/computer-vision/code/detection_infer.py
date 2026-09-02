"""Object detection inference with a pretrained torchvision detector — CPU only.

Companion script for:
  Machine Learning/Worked Examples/computer-vision/object-detection-coco.md

Environment (SPEC-ML-5 / NOTE-ML-5, shared ML virtualenv `.venv-ml`):
  torch==2.14.0+cpu
  torchvision==0.29.0+cpu
  matplotlib==3.11.1
  numpy==2.5.2

Run with the ML venv's interpreter, from anywhere (paths below are resolved relative to
this file, not to the current working directory):

    .venv-ml/Scripts/python.exe "Machine Learning/Worked Examples/computer-vision/code/detection_infer.py"

First run downloads:
  - the Faster R-CNN COCO weights (~160 MB) via torchvision into the torch hub cache
    (`~/.cache/torch/hub/checkpoints/`, or `%USERPROFILE%\\.cache\\torch\\hub\\checkpoints\\` on
    Windows) — this is torchvision's own cache, outside the repo, so it is never committed;
  - two sample photos (~90-190 KB each) from the official torchvision example gallery into
    ./datasets/_downloaded/detection/ next to this file (gitignored).

This script does NOT train anything — SPEC-ML-5 scopes training a detector on full COCO out of
this chapter (conceptual only). Every box drawn below comes from running the pretrained model
forward on a real image; nothing here is a hand-authored or fabricated detection.
"""
from __future__ import annotations

import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torchvision.io import decode_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

# --------------------------------------------------------------------------------------
# Reproducibility — same habit as image-classification-mnist.md's script. Inference through
# a frozen, pretrained model is far less randomness-sensitive than training (no weight
# initialisation, no shuffling), but NMS tie-breaking on exactly-equal scores is still, in
# principle, order-dependent, so every RNG is fixed anyway.
# --------------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths are resolved relative to this file so the script runs the same way regardless of
# the shell's current directory.
CHAPTER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CHAPTER_DIR / "code" / "datasets" / "_downloaded" / "detection"
ARTEFACTS_DIR = CHAPTER_DIR / "artefacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample images: two photos from torchvision's own example gallery — the same directory whose
# "dog1.jpg"/"dog2.jpg" the official visualization-utils example loads with
# `Path('../assets') / 'dog1.jpg'` to demonstrate fasterrcnn_resnet50_fpn (source:
# https://docs.pytorch.org/vision/stable/auto_examples/others/plot_visualization_utils.html,
# checked 2026-09-02; NOTE-ML-5 evidence #7 independently grounds this gallery directory as a
# COCO-style sample-image source). "person1.jpg" and "leaning_tower.jpg" are used here instead
# of the single-dog photos because they contain multiple, different real-world objects —
# better for showing "what is where" than one dominant animal filling the frame. Repository
# licence: BSD-3-Clause (https://github.com/pytorch/vision/blob/main/LICENSE, checked
# 2026-09-02), the same terms under which PyTorch's own docs reuse these files.
SAMPLE_IMAGE_BASE_URL = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets"
SAMPLE_IMAGE_NAMES = ["person1.jpg", "leaning_tower.jpg"]

# Post-processing knobs (SPEC-ML-5 LO3): a confidence floor and an NMS IoU threshold, both
# applied explicitly in this script even though Faster R-CNN already runs its own internal
# NMS before returning scores (NOTE-ML-5: "output scores are post-NMS"). Explicit is worth
# doing anyway — Section 4 of the chapter explains why, and a raw one-stage detector's dense
# per-anchor output would need this step for real, not just as an illustration.
SCORE_THRESHOLD = 0.7
NMS_IOU_THRESHOLD = 0.5


def download_sample_images() -> list[Path]:
    """Download the two sample photos into DATA_DIR if not already cached; return their paths."""
    paths = []
    for name in SAMPLE_IMAGE_NAMES:
        dest = DATA_DIR / name
        if not dest.exists():
            url = f"{SAMPLE_IMAGE_BASE_URL}/{name}"
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed, trusted URL
        paths.append(dest)
    return paths


def load_detector() -> tuple[torch.nn.Module, FasterRCNN_ResNet50_FPN_Weights, list[str]]:
    """Load the pretrained Faster R-CNN detector and its COCO category names.

    `weights.meta["categories"]` returns a 91-entry list (verified by inspecting it directly
    below and printed in the run log): index 0 is "__background__", and indices 1-90 mirror
    the *original* COCO detection-challenge category-id numbering, which has 10 unused ids
    ("N/A" placeholders) mixed in among the real 80 classes. `outputs["labels"]` values index
    directly into this same 91-entry list, so `categories[label]` is always correct without
    any manual offset — indexing with `label - 1` (a very natural first guess) is wrong and is
    covered as a pitfall in the chapter.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    categories = weights.meta["categories"]
    print(f"Weights: {weights} (box_map on COCO-val2017: "
          f"{weights.meta['_metrics']['COCO-val2017']['box_map']})")
    print(f"Category list length: {len(categories)} "
          f"('N/A' placeholders: {sum(1 for c in categories if c == 'N/A')})")

    model = fasterrcnn_resnet50_fpn(weights=weights, progress=True)
    model = model.to(DEVICE)
    model.eval()  # inference mode: disables dropout/batchnorm training behaviour
    return model, weights, categories


def run_inference(
    model: torch.nn.Module,
    weights: FasterRCNN_ResNet50_FPN_Weights,
    image_path: Path,
) -> tuple[torch.Tensor, dict]:
    """Run the detector on one image; return (original uint8 image tensor, raw output dict)."""
    image_uint8 = decode_image(str(image_path))  # (C, H, W) uint8, RGB
    preprocess = weights.transforms()  # the exact resize/normalise this model was trained with
    batch = [preprocess(image_uint8).to(DEVICE)]
    with torch.no_grad():
        outputs = model(batch)
    return image_uint8, outputs[0]


def postprocess(
    output: dict, score_threshold: float, nms_iou_threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a score threshold, then explicit NMS; return (boxes, labels, scores) kept."""
    boxes, labels, scores = output["boxes"], output["labels"], output["scores"]

    keep_score = scores >= score_threshold
    boxes, labels, scores = boxes[keep_score], labels[keep_score], scores[keep_score]

    # torchvision.ops.nms: "removes boxes which have an IoU greater than iou_threshold with
    # another (higher scoring) box" (source: NOTE-ML-5 API grounding + torchvision.ops docs,
    # https://docs.pytorch.org/vision/0.29/generated/torchvision.ops.nms.html, checked
    # 2026-09-02). Kept mostly to make the mechanism visible; Faster R-CNN's own internal NMS
    # already removed most duplicate boxes before `scores` ever reached this function.
    keep_nms = nms(boxes, scores, nms_iou_threshold)
    return boxes[keep_nms], labels[keep_nms], scores[keep_nms]


def draw_and_save(
    image_uint8: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    categories: list[str],
    out_path: Path,
) -> None:
    label_strings = [f"{categories[label]}: {score:.2f}" for label, score in zip(labels.tolist(), scores.tolist())]
    annotated = draw_bounding_boxes(
        image_uint8,
        boxes=boxes,
        labels=label_strings,
        colors="red",
        width=3,
    )
    pil_image = to_pil_image(annotated)
    # Cap the artefact's longest side at 1200px purely to keep the committed PNG small — this
    # happens strictly after drawing, so the boxes above were computed and rendered against the
    # model's full-resolution input; only the saved file is downscaled.
    max_side = max(pil_image.size)
    if max_side > 1200:
        scale = 1200 / max_side
        new_size = (round(pil_image.width * scale), round(pil_image.height * scale))
        pil_image = pil_image.resize(new_size)
    pil_image.save(out_path)


def format_detections_table(rows: list[dict]) -> str:
    header = "| image | label | score | box (x1, y1, x2, y2) |\n|---|---|---|---|\n"
    lines = [
        f"| {r['image']} | {r['label']} | {r['score']:.3f} "
        f"| ({r['box'][0]:.0f}, {r['box'][1]:.0f}, {r['box'][2]:.0f}, {r['box'][3]:.0f}) |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def main() -> None:
    print(f"Device: {DEVICE}")
    image_paths = download_sample_images()
    model, weights, categories = load_detector()

    all_rows: list[dict] = []
    start = time.perf_counter()
    for image_path in image_paths:
        image_uint8, raw_output = run_inference(model, weights, image_path)
        print(f"\n{image_path.name}: {raw_output['boxes'].shape[0]} raw detections "
              f"(all confidence levels, pre-threshold)")

        boxes, labels, scores = postprocess(raw_output, SCORE_THRESHOLD, NMS_IOU_THRESHOLD)
        print(f"{image_path.name}: {boxes.shape[0]} detections kept "
              f"(score >= {SCORE_THRESHOLD}, NMS IoU <= {NMS_IOU_THRESHOLD})")

        out_name = f"detection_{image_path.stem}.png"
        draw_and_save(image_uint8, boxes, labels, scores, categories, ARTEFACTS_DIR / out_name)
        print(f"Wrote {ARTEFACTS_DIR / out_name}")

        for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            all_rows.append({
                "image": image_path.name,
                "label": categories[label],
                "score": score,
                "box": box,
            })

    elapsed = time.perf_counter() - start
    print(f"\nTotal inference wall-clock ({len(image_paths)} images): {elapsed:.2f}s on {DEVICE}")

    table_md = format_detections_table(all_rows)
    table_path = ARTEFACTS_DIR / "detections_table.md"
    table_path.write_text(table_md, encoding="utf-8")
    print(f"\nDetections table written to {table_path}:\n")
    print(table_md)


if __name__ == "__main__":
    main()
