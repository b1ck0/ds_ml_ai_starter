# NOTE-ML-7: Computer Vision Metrics (IoU, AP, mAP, mAR)

**Answer:** TorchMetrics 1.9.0 `MeanAveragePrecision` (detection) accepts predictions/targets as lists of dicts with keys: 'boxes' (Nx4 FloatTensor, xyxy or xywh format), 'scores' (N FloatTensor, [0,1]), 'labels' (N LongTensor, class indices). Returns dict with 'map' (mAP@[.5:.95]), 'map_50' (AP@.50, Pascal VOC style), 'map_75' (AP@.75), 'mar_1', 'mar_10', 'mar_100', and 'mar_small', 'mar_medium', 'mar_large' (recall by object size). **IoU definition:** (intersection area) / (union area) for two boxes or masks, ∈ [0,1]. **AP (Average Precision):** area under precision-recall curve; COCO uses 101-point interpolation (recall divided into 101 equally-spaced points, precision at each point is max precision to the right). **COCO mAP@[.5:.95]:** average of APs computed at 10 IoU thresholds [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], averaged over all 80 categories. **mAR:** mean Average Recall (same structure as mAP but recall-based). Checked 2026-09-02.

**Evidence:**

1. **TorchMetrics 1.9.0 MeanAveragePrecision API** (verified 2026-09-02)
   - Source: https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html (v1.9.0)
   - Input format (predictions): `[{'boxes': Tensor(N,4), 'scores': Tensor(N), 'labels': Tensor(N)}, ...]`
   - Input format (targets): same structure
   - Supported box formats: 'xyxy' (x1, y1, x2, y2) or 'xywh' (x, y, width, height)
   - Return value: OrderedDict with keys 'map', 'map_50', 'map_75', 'mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large'

2. **IoU (Intersection over Union) definition** (verified 2026-09-02)
   - Formula: IoU(A, B) = |A ∩ B| / |A ∪ B|
   - Range: [0, 1] where 0 = no overlap, 1 = perfect overlap
   - Used to match predicted boxes to ground-truth boxes
   - Source: "Mean Average Precision (mAP) in Object Detection" https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
   - Widely accepted metric across COCO, PASCAL VOC, and other detection benchmarks

3. **Average Precision (AP) definition** (verified 2026-09-02)
   - AP = area under the Precision-Recall (PR) curve for a single class at a fixed IoU threshold
   - **11-point interpolation (PASCAL VOC):** average precision at 11 recall levels [0.0, 0.1, 0.2, ..., 1.0]
   - **All-point / 101-point interpolation (COCO):** recall divided into 101 equally-spaced points; at each point, precision = max precision to the right of that recall value; AP = average of these 101 precision values
   - Source: "A Comprehensive Guide to Mean Average Precision" https://www.lightly.ai/blog/mean-average-precision
   - Quote: "In COCO mAP, an average for a 101-point interpolated AP is calculated"

4. **COCO mAP@[.5:.95] definition** (verified 2026-09-02)
   - **IoU threshold range:** 10 thresholds: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
   - **Calculation:** For each threshold τ, compute AP@τ for each category; average across τ and categories
   - **Step size:** 0.05 between thresholds
   - **Formula:** mAP@[.5:.95] = (1/10) * Σ(τ=0.5 to 0.95, Δ=0.05) [ (1/80) * Σ(c=1 to 80) AP(c, τ) ]
   - Source: https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation
   - Quote: "For the COCO 2017 challenge, the mAP was calculated by averaging the AP over all 80 object categories AND all 10 IoU thresholds from 0.50 to 0.95"

5. **COCO variant metrics** (verified 2026-09-02)
   - **AP@.50** (PASCAL VOC-style): single IoU threshold at 0.50; easier metric
   - **AP@.75:** IoU threshold at 0.75; stricter metric
   - **AP_small, AP_medium, AP_large:** AP broken down by object size (area < 32², 32² to 96², > 96² pixels)

6. **mAR (mean Average Recall)** (verified 2026-09-02)
   - Similar structure to mAP but focuses on recall instead of precision
   - mAR@[.5:.95] = average recall across the same 10 IoU thresholds
   - Often reported as mAR@100 (max 100 detections per image)
   - Complements mAP to assess false negatives

7. **TP/FP/FN matching** (verified 2026-09-02)
   - Prediction matched to ground-truth box if IoU ≥ threshold → True Positive (TP)
   - Prediction not matched or IoU < threshold → False Positive (FP)
   - Ground-truth box not matched by any prediction → False Negative (FN)
   - Precision = TP / (TP + FP); Recall = TP / (TP + FN)

**Caveats / limits:**

- **101-point vs 11-point:** COCO uses 101-point (all-point) interpolation; older Pascal VOC used 11-point. Code must use correct interpolation method.
- **Class imbalance:** COCO has imbalanced class sizes (people >> toasters); mAP treats all classes equally, so rare classes matter. Segmentation has pixel imbalance (background >> foreground).
- **IoU threshold sensitivity:** AP is very sensitive to IoU threshold; small boxes harder to achieve high IoU.
- **NMS threshold:** Post-processing NMS affects metrics; different NMS thresholds produce different AP/mAR values.
- **Version sensitivity:** torchmetrics 1.9.0 API confirmed; earlier versions (<0.11) had different interfaces.

**Recommendation:**

1. **For chapter:** Define IoU, TP/FP/FN, and PR curve before introducing AP; use visualization (Venn diagram for IoU, PR curve plot).
2. **Hand-compute AP on toy example (≤5 predictions + 5 ground-truth):** Show the 101-point interpolation explicitly so readers see where the numbers come from.
3. **Verify torchmetrics results against hand computation:** Build trust that the library is correct.
4. **Clarify COCO-specific choices:** Why 10 thresholds? Why 80 classes? Why size-based splits?
5. **For segmentation (reference ML-6):** Similar metrics (mIoU) but computed per-pixel instead of per-box.
6. **Example code:** Use the `MeanAveragePrecision(iou_type="bbox")` constructor and call it as a metric object with update/compute pattern.
