# SPEC-ML-7: CV Metrics — IoU, mAP, mAR explained and computed

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Worked Examples (Computer Vision)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-6 (precision/recall), SPEC-ML-5 (detection)

## Intent
The metrics that judge detection/segmentation are their own topic. Teach IoU from first principles,
then how precision/recall become a Precision-Recall curve per class, then mAP and mAR — computed on a
small, fully-runnable example so the numbers are transparent.

## Learning objectives
- LO1 — Define IoU (intersection over union) and compute it for boxes and masks by hand.
- LO2 — Explain how an IoU threshold turns detections into TP/FP/FN, and build a per-class PR curve.
- LO3 — Define Average Precision (AP), mean AP (mAP) incl. the COCO mAP@[.5:.95], and mean Average Recall (mAR).
- LO4 — Use a metrics library (torchmetrics) to reproduce the values and trust-but-verify against the hand computation.

## Scope
In: IoU (boxes+masks), TP/FP/FN via IoU threshold, PR curve, AP/mAP/mAR, COCO-style averaging, torchmetrics.
Out: full COCO eval harness internals (link), segmentation-specific panoptic quality (mention).

## Outline
1. What & why — why classification metrics need an IoU notion for localisation.
2. IoU by hand — two boxes; then two masks; visualise the overlap.
3. From IoU to TP/FP/FN — matching, the threshold; a small worked set of detections.
4. PR curve → AP → mAP (and COCO mAP@[.5:.95]); mAR.
5. Reproduce with torchmetrics; compare to the hand numbers.
6. Pitfalls — mixing up mAP definitions, IoU-threshold sensitivity, class imbalance.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/computer-vision/cv-metrics.md"
- Code: "Machine Learning/Worked Examples/computer-vision/code/cv_metrics.py"
- Artefacts: an IoU illustration; a PR curve; a table comparing hand-computed vs torchmetrics AP/mAP.

## Claims to ground (Haiku, before writing)
- [ ] Verify the exact definitions of IoU, AP (11-point / all-point interpolation), COCO mAP@[.5:.95], and mAR against authoritative sources (COCO eval / torchmetrics docs).
- [ ] Verify current torchmetrics version + the detection metric API (MeanAveragePrecision) and its input format.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — cv_metrics.py computes IoU + AP/mAP by hand AND via torchmetrics on a small example, numbers agree, artefacts produced; snippet-check passes. AC3 — metric definitions + torchmetrics API grounded. AC4 — every metric derived, not just named; ties back to precision/recall from DS-6.

## Gates
Entry: approved; notes landed. Exit: DoD checklist. Uses .venv-ml (+ torchmetrics).
