"""CV metrics from first principles: IoU -> TP/FP/FN -> PR curve -> AP/mAP/mAR.

SPEC-ML-7 / NOTE-ML-6-cv-metrics.md.

Everything here is computed twice: once by hand (plain Python, no detection library) and once with
torchmetrics' MeanAveragePrecision, on the SAME small synthetic detection set, so the two numbers can
be compared directly. Run with the .venv-ml interpreter:

    .venv-ml/Scripts/python.exe "Machine Learning/Worked Examples/computer-vision/code/cv_metrics.py"

Environment (verified against the installed .venv-ml on 2026-09-02):
    torch==2.14.0+cpu
    torchmetrics==1.9.0
    faster-coco-eval==1.8.0   (backend for torchmetrics[detection]; see Section 5 pitfall)
    numpy==2.5.2
    matplotlib==3.11.1
    Python 3.13.7
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = Path(__file__).resolve().parent
ARTEFACTS = HERE.parent / "artefacts"
ARTEFACTS.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "cat", 1: "dog"}
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95, COCO's 10 thresholds


# ---------------------------------------------------------------------------
# 1. IoU from first principles — boxes
# ---------------------------------------------------------------------------

def iou_box(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union of two axis-aligned boxes in (x1, y1, x2, y2) format.

    IoU(A, B) = area(A ∩ B) / area(A ∪ B)   [source: NOTE-ML-6-cv-metrics.md, item 2].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. IoU from first principles — masks
# ---------------------------------------------------------------------------

def iou_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-Union of two boolean pixel masks of the same shape."""
    assert mask_a.shape == mask_b.shape, "masks must share a shape to compare pixel-for-pixel"
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def make_rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask of `shape` with `rect` = (row0, col0, row1, col1) set True."""
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0:r1, c0:c1] = True
    return mask


# ---------------------------------------------------------------------------
# 3. Illustrate IoU (boxes + masks) as an artefact
# ---------------------------------------------------------------------------

def plot_iou_illustration(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # --- boxes ---
    box_gt = (10, 10, 50, 50)
    box_pred = (30, 30, 70, 70)
    iou_val = iou_box(box_gt, box_pred)

    ax = axes[0]
    ax.add_patch(mpatches.Rectangle((box_gt[0], box_gt[1]), box_gt[2] - box_gt[0], box_gt[3] - box_gt[1],
                                     fill=False, edgecolor="tab:green", linewidth=2, label="ground truth"))
    ax.add_patch(mpatches.Rectangle((box_pred[0], box_pred[1]), box_pred[2] - box_pred[0],
                                     box_pred[3] - box_pred[1],
                                     fill=False, edgecolor="tab:orange", linewidth=2, label="prediction"))
    ix1, iy1 = max(box_gt[0], box_pred[0]), max(box_gt[1], box_pred[1])
    ix2, iy2 = min(box_gt[2], box_pred[2]), min(box_gt[3], box_pred[3])
    ax.add_patch(mpatches.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1,
                                     facecolor="tab:red", alpha=0.35, label="intersection"))
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"Box IoU = {iou_val:.3f}")
    ax.legend(loc="upper right", fontsize=8)

    # --- masks ---
    shape = (10, 10)
    mask_gt = make_rect_mask(shape, (2, 2, 8, 8))     # rows 2-7, cols 2-7 -> 36 px
    mask_pred = make_rect_mask(shape, (4, 4, 10, 10))  # rows 4-9, cols 4-9 -> 36 px
    m_iou = iou_mask(mask_gt, mask_pred)

    overlay = np.zeros((*shape, 3))
    overlay[mask_gt & ~mask_pred] = (0.0, 0.6, 0.2)     # green: GT only
    overlay[mask_pred & ~mask_gt] = (0.9, 0.5, 0.0)     # orange: pred only
    overlay[mask_gt & mask_pred] = (0.8, 0.1, 0.1)      # red: intersection

    ax = axes[1]
    ax.imshow(overlay, origin="upper")
    ax.set_title(f"Mask IoU = {m_iou:.3f}  ({int((mask_gt & mask_pred).sum())} px ∩ / "
                 f"{int((mask_gt | mask_pred).sum())} px ∪)")
    ax.set_xticks(range(shape[1]))
    ax.set_yticks(range(shape[0]))
    ax.grid(color="white", linewidth=0.5)
    legend_handles = [
        mpatches.Patch(color=(0.0, 0.6, 0.2), label="ground truth only"),
        mpatches.Patch(color=(0.9, 0.5, 0.0), label="prediction only"),
        mpatches.Patch(color=(0.8, 0.1, 0.1), label="intersection"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, bbox_to_anchor=(1.05, -0.05))

    fig.suptitle("Intersection over Union — boxes (left) and masks (right)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[iou] box example IoU = {iou_val:.4f}  (boxes {box_gt} vs {box_pred})")
    print(f"[iou] mask example IoU = {m_iou:.4f}  (10x10 grid, two 6x6 rectangles)")


# ---------------------------------------------------------------------------
# 4. A small, hand-checkable detection set (2 images, 2 classes)
# ---------------------------------------------------------------------------
# Ground truth: image_id, class_id, box (x1, y1, x2, y2)
GROUND_TRUTH = [
    (0, 0, (0, 0, 40, 40)),      # G0 cat, image 0
    (0, 1, (50, 0, 90, 40)),     # G1 dog, image 0
    (1, 0, (10, 10, 50, 50)),    # G2 cat, image 1
    (1, 1, (60, 60, 100, 100)),  # G3 dog, image 1
    (1, 0, (0, 60, 50, 100)),    # G4 cat, image 1 (partially detected -> shows threshold sensitivity)
]

# Predictions: image_id, class_id, box, score
PREDICTIONS = [
    (0, 0, (0, 0, 40, 40), 0.95),      # P0 cat, exact match of G0        -> IoU 1.0
    (0, 0, (30, 30, 70, 70), 0.55),    # P1 cat, barely overlaps G0        -> IoU ~0.032
    (0, 1, (50, 0, 90, 40), 0.85),     # P2 dog, exact match of G1        -> IoU 1.0
    (1, 0, (10, 10, 50, 50), 0.75),    # P3 cat, exact match of G2        -> IoU 1.0
    (1, 1, (0, 60, 20, 80), 0.30),     # P4 dog, no overlap with G3       -> IoU 0.0
    (1, 0, (5, 65, 45, 105), 0.65),    # P5 cat, partial overlap with G4  -> IoU ~0.636
]


# ---------------------------------------------------------------------------
# 5. IoU threshold -> TP/FP/FN, greedy match, one class at a time
# ---------------------------------------------------------------------------

def match_class(preds_cls: list[tuple], gts_cls: list[tuple], iou_thr: float):
    """Greedy TP/FP assignment for one class, COCO/VOC style.

    preds_cls: list of (image_id, box, score); gts_cls: list of (image_id, box).
    Detections are processed highest-score-first; each ground-truth box can be
    claimed by at most one detection (its best-IoU match above the threshold).
    Returns: is_tp (bool array aligned with preds sorted by score desc), sorted scores,
    num_gt (int).
    """
    order = sorted(range(len(preds_cls)), key=lambda i: preds_cls[i][2], reverse=True)
    claimed = [False] * len(gts_cls)
    is_tp = []
    scores_sorted = []
    for i in order:
        img_id, box, score = preds_cls[i]
        best_iou, best_j = 0.0, -1
        for j, (g_img, g_box) in enumerate(gts_cls):
            if g_img != img_id or claimed[j]:
                continue
            iou = iou_box(box, g_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            claimed[best_j] = True
            is_tp.append(True)
        else:
            is_tp.append(False)
        scores_sorted.append(score)
    return np.array(is_tp), np.array(scores_sorted), len(gts_cls)


def precision_recall_curve(is_tp: np.ndarray, num_gt: int):
    """Cumulative precision/recall at each rank, in score-descending order."""
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(~is_tp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / num_gt if num_gt > 0 else np.zeros_like(tp_cum, dtype=float)
    return precision, recall


def ap_coco_101point(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO-style 101-point interpolated Average Precision.

    Recall is sampled at 101 equally spaced points [0, 0.01, ..., 1.0]; at each point the
    interpolated precision is the MAX precision observed at any recall >= that point (the
    "precision envelope"). AP is the mean of those 101 interpolated precision values.
    [source: NOTE-ML-6-cv-metrics.md, item 3 — COCO 101-point interpolation.]
    """
    recall_levels = np.linspace(0.0, 1.0, 101)
    if len(precision) == 0:
        return 0.0
    interpolated = np.zeros(101)
    for k, r in enumerate(recall_levels):
        mask = recall >= r
        interpolated[k] = precision[mask].max() if mask.any() else 0.0
    return float(interpolated.mean())


def hand_compute(iou_thr: float):
    """Per-class AP and recall (all detections, no interpolation) at one IoU threshold."""
    results = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        preds_cls = [(img, box, score) for img, c, box, score in PREDICTIONS if c == cls_id]
        gts_cls = [(img, box) for img, c, box in GROUND_TRUTH if c == cls_id]
        is_tp, scores_sorted, num_gt = match_class(preds_cls, gts_cls, iou_thr)
        precision, recall = precision_recall_curve(is_tp, num_gt)
        ap = ap_coco_101point(precision, recall)
        ar = float(recall[-1]) if len(recall) > 0 else 0.0  # recall using ALL detections, no interpolation
        results[cls_name] = {
            "ap": ap, "ar": ar, "precision": precision, "recall": recall,
            "scores": scores_sorted, "is_tp": is_tp, "num_gt": num_gt,
        }
    return results


# ---------------------------------------------------------------------------
# 6. PR-curve artefact (IoU = 0.5)
# ---------------------------------------------------------------------------

def plot_pr_curves(results_at_50: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {"cat": "tab:blue", "dog": "tab:orange"}
    for cls_name, res in results_at_50.items():
        recall, precision = res["recall"], res["precision"]
        # step curve: prepend the (0, first-precision) point for a clean plot start
        r_plot = np.concatenate([[0.0], recall])
        p_plot = np.concatenate([[precision[0] if len(precision) else 1.0], precision])
        ax.step(r_plot, p_plot, where="post", label=f"{cls_name} (AP={res['ap']:.3f})",
                 color=colors[cls_name], linewidth=2)
        ax.scatter(recall, precision, color=colors[cls_name], zorder=3, s=25)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class Precision-Recall curve (IoU threshold = 0.50)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. torchmetrics MeanAveragePrecision on the same data
# ---------------------------------------------------------------------------

def torchmetrics_compute():
    image_ids = sorted({img for img, *_ in GROUND_TRUTH} | {img for img, *_ in PREDICTIONS})

    preds = []
    targets = []
    for img_id in image_ids:
        p_boxes = [box for i, c, box, s in PREDICTIONS if i == img_id]
        p_scores = [s for i, c, box, s in PREDICTIONS if i == img_id]
        p_labels = [c for i, c, box, s in PREDICTIONS if i == img_id]
        preds.append({
            "boxes": torch.tensor(p_boxes, dtype=torch.float32) if p_boxes else torch.zeros((0, 4)),
            "scores": torch.tensor(p_scores, dtype=torch.float32),
            "labels": torch.tensor(p_labels, dtype=torch.int64),
        })

        g_boxes = [box for i, c, box in GROUND_TRUTH if i == img_id]
        g_labels = [c for i, c, box in GROUND_TRUTH if i == img_id]
        targets.append({
            "boxes": torch.tensor(g_boxes, dtype=torch.float32) if g_boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(g_labels, dtype=torch.int64),
        })

    # backend="faster_coco_eval": this .venv-ml has faster-coco-eval installed, not pycocotools
    # (torchmetrics' default backend). See the chapter's "pitfalls" section for why this matters.
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,
                                   backend="faster_coco_eval")
    metric.update(preds, targets)
    result = metric.compute()
    return result


# ---------------------------------------------------------------------------
# 8. Comparison table artefact
# ---------------------------------------------------------------------------

def write_comparison_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["metric", "hand_computed", "torchmetrics", "abs_diff"]
    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join(str(row[f]) for f in fieldnames))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print(f"IoU thresholds used for mAP@[.5:.95]: {IOU_THRESHOLDS}")

    # --- Section 3: IoU illustration ---
    plot_iou_illustration(ARTEFACTS / "iou_illustration.png")

    # --- Sections 5-6: hand AP@0.5, PR curve ---
    results_50 = hand_compute(0.50)
    print("\n=== Hand-computed, IoU threshold = 0.50 ===")
    for cls_name, res in results_50.items():
        print(f"  {cls_name}: is_tp={res['is_tp'].tolist()} scores={res['scores'].tolist()} "
              f"num_gt={res['num_gt']} -> AP={res['ap']:.4f}  AR(all-dets)={res['ar']:.4f}")
    plot_pr_curves(results_50, ARTEFACTS / "pr_curve_iou50.png")

    # --- Section 4/5: full COCO-style mAP@[.5:.95] and mAR by hand ---
    per_class_ap = {name: [] for name in CLASS_NAMES.values()}
    per_class_ar = {name: [] for name in CLASS_NAMES.values()}
    for thr in IOU_THRESHOLDS:
        res = hand_compute(thr)
        for cls_name, r in res.items():
            per_class_ap[cls_name].append(r["ap"])
            per_class_ar[cls_name].append(r["ar"])

    print("\n=== Hand-computed AP per class per IoU threshold ===")
    for cls_name in CLASS_NAMES.values():
        formatted = [f"{t}:{a:.3f}" for t, a in zip(IOU_THRESHOLDS, per_class_ap[cls_name])]
        print(f"  {cls_name}: {formatted}")

    hand_ap50 = {name: per_class_ap[name][0] for name in CLASS_NAMES.values()}  # thr list starts at 0.50
    hand_ap_mean = {name: float(np.mean(per_class_ap[name])) for name in CLASS_NAMES.values()}
    hand_ar_mean = {name: float(np.mean(per_class_ar[name])) for name in CLASS_NAMES.values()}

    hand_map50 = float(np.mean(list(hand_ap50.values())))
    hand_map_5095 = float(np.mean(list(hand_ap_mean.values())))
    hand_mar_5095 = float(np.mean(list(hand_ar_mean.values())))

    print(f"\nhand mAP@0.50       = {hand_map50:.4f}")
    print(f"hand mAP@[.5:.95]    = {hand_map_5095:.4f}")
    print(f"hand mAR@[.5:.95]    = {hand_mar_5095:.4f}")

    # --- Section 7: torchmetrics ---
    tm_result = torchmetrics_compute()
    print("\n=== torchmetrics MeanAveragePrecision.compute() ===")
    for k, v in tm_result.items():
        print(f"  {k}: {v}")

    tm_map50 = float(tm_result["map_50"])
    tm_map_5095 = float(tm_result["map"])
    tm_mar100 = float(tm_result["mar_100"])

    # per-class AP@[.5:.95] from torchmetrics (class_metrics=True), mapped id -> name
    tm_classes = tm_result["classes"].tolist()
    tm_map_per_class = tm_result["map_per_class"].tolist()
    tm_map_per_class_named = {CLASS_NAMES[c]: v for c, v in zip(tm_classes, tm_map_per_class)}

    # --- comparison table ---
    def diff(a, b):
        return abs(a - b)

    rows = [
        {"metric": "mAP@0.50", "hand_computed": round(hand_map50, 4),
         "torchmetrics": round(tm_map50, 4), "abs_diff": round(diff(hand_map50, tm_map50), 4)},
        {"metric": "mAP@[.5:.95]", "hand_computed": round(hand_map_5095, 4),
         "torchmetrics": round(tm_map_5095, 4), "abs_diff": round(diff(hand_map_5095, tm_map_5095), 4)},
        {"metric": "mAR@[.5:.95] (all-dets, hand) vs mar_100 (torchmetrics)",
         "hand_computed": round(hand_mar_5095, 4),
         "torchmetrics": round(tm_mar100, 4), "abs_diff": round(diff(hand_mar_5095, tm_mar100), 4)},
    ]
    for cls_name in CLASS_NAMES.values():
        rows.append({
            "metric": f"AP@[.5:.95] class={cls_name}",
            "hand_computed": round(hand_ap_mean[cls_name], 4),
            "torchmetrics": round(tm_map_per_class_named[cls_name], 4),
            "abs_diff": round(diff(hand_ap_mean[cls_name], tm_map_per_class_named[cls_name]), 4),
        })

    print("\n=== Hand vs torchmetrics comparison ===")
    for row in rows:
        print(f"  {row}")

    write_comparison_csv(rows, ARTEFACTS / "hand_vs_torchmetrics.csv")

    print(f"\nArtefacts written to: {ARTEFACTS}")


if __name__ == "__main__":
    main()
