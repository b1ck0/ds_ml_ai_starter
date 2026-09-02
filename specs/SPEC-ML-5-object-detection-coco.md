# SPEC-ML-5: Object Detection — pretrained detectors on COCO classes

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Worked Examples (Computer Vision)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-4, SPEC-ML-7 (metrics — can be read alongside)
**Nature:** RUNNABLE INFERENCE — training a detector on full COCO can't run in a sandbox. We run a
torchvision PRETRAINED detector (COCO weights) on sample images for real inference + visualisation.

## Intent
Move from "what is in this image" to "what is where". Use a torchvision pretrained detector to draw
real bounding boxes on sample images, explain the model family, and connect to the detection metrics.

## Learning objectives
- LO1 — Explain object detection vs classification (boxes + labels + scores) and the two-stage vs one-stage families.
- LO2 — Load a torchvision pretrained detector (e.g. Faster R-CNN / RetinaNet / SSD) and run inference on images.
- LO3 — Interpret outputs (boxes, labels, scores), apply a score threshold and NMS, and visualise.
- LO4 — Connect to evaluation (IoU, mAP — detailed in ML-7) and the COCO dataset/label set.

## Scope
In: pretrained detector inference on a few sample images, thresholding/NMS, visualisation, COCO label map, model-family overview.
Out: training/fine-tuning a detector (conceptual only), full COCO download.

## Outline
1. What & why — detection as classification+localisation; where it's used.
2. Model families — two-stage (Faster R-CNN) vs one-stage (SSD/RetinaNet/YOLO-style); speed/accuracy trade-off.
3. Run a pretrained detector on sample images; boxes + labels + scores.
4. Post-processing — score threshold, NMS; visualise the boxes.
5. Evaluation preview — IoU + mAP (→ ML-7); COCO's 80 classes.
6. Pitfalls — wrong preprocessing/normalisation, ignoring score threshold, class-index off-by-one.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/computer-vision/object-detection-coco.md"
- Code: "Machine Learning/Worked Examples/computer-vision/code/detection_infer.py"
- Artefacts: input images with drawn bounding boxes; a detections table (label, score, box).

## Claims to ground (Haiku, before writing)
- [ ] Verify current torchvision detection API: the `torchvision.models.detection` model + the `weights=` enum API (e.g. FasterRCNN_ResNet50_FPN_Weights) and how to get the COCO category names from the weights meta. Confirm weights download works.
- [ ] Verify a source of freely-usable sample images (torchvision sample / a CC-licensed image) to run on.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — detection_infer.py RUNS real inference with pretrained weights on sample image(s) and draws real boxes; snippet-check passes; no fabricated detections. AC3 — torchvision detection API + weights + sample-image source grounded. AC4 — detection-vs-classification made clear; training-is-out-of-scope stated honestly.

## Gates
Entry: approved; notes landed. Exit: DoD checklist. Uses .venv-ml.
