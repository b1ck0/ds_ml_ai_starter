# SPEC-ML-6: Semantic Segmentation — pretrained models, pixel-wise labels

**Status:** approved
**Subject:** Machine Learning
**Section:** Worked Examples (Computer Vision)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-5, SPEC-ML-7
**Nature:** RUNNABLE INFERENCE — use a torchvision PRETRAINED segmentation model on sample images.

## Intent
Go from boxes to pixels: assign every pixel a class. Run a pretrained segmentation model, visualise
the masks, and explain how segmentation differs from detection and how it's evaluated (mIoU).

## Learning objectives
- LO1 — Distinguish semantic vs instance vs panoptic segmentation; here focus on semantic (per-pixel class).
- LO2 — Load a torchvision pretrained segmentation model (e.g. DeepLabV3 / FCN) and run inference.
- LO3 — Turn the output logits into a class mask and overlay it on the image.
- LO4 — Explain segmentation metrics (pixel accuracy, mIoU) and their pitfalls (class imbalance in pixels).

## Scope
In: pretrained semantic-segmentation inference, mask visualisation/overlay, metric intuition.
Out: training, instance/panoptic depth (mention), full dataset download.

## Outline
1. What & why — pixel-level understanding; use cases (medical, autonomous driving).
2. Semantic vs instance vs panoptic — one picture each.
3. Run a pretrained model; argmax the logits to a mask; overlay.
4. Metrics — pixel accuracy vs mIoU; why mIoU is fairer.
5. Pitfalls — resolution/normalisation, tiny-class suppression, reading logits vs probs.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/computer-vision/semantic-segmentation-coco.md"
- Code: "Machine Learning/Worked Examples/computer-vision/code/segmentation_infer.py"
- Artefacts: original image + colourised mask overlay; a per-class legend.

## Claims to ground (Haiku, before writing)
- [ ] Verify the current torchvision segmentation API: `torchvision.models.segmentation` model + weights enum (e.g. DeepLabV3_ResNet50_Weights / FCN_ResNet50_Weights), output dict shape ('out'), and the class list from weights meta. Confirm weights download.
- [ ] Reuse ML-5's sample-image source.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — segmentation_infer.py RUNS real inference and produces a real mask overlay; snippet-check passes; no fabricated masks. AC3 — torchvision segmentation API + weights grounded. AC4 — detection-vs-segmentation contrast clear; mIoU intuition given (formal IoU in ML-7).

## Gates
Entry: approved; notes landed. Exit: DoD checklist. Uses .venv-ml.
