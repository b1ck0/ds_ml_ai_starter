# SPEC-ML-4: Image Classification — MNIST with PyTorch

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Worked Examples (Computer Vision)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-0 (env), SPEC-ML-1/ML-2 (theory)

## Intent
The "hello world" of deep learning, fully runnable on CPU. Train a small CNN on MNIST with PyTorch +
torchvision, end to end: data → model → training loop → evaluation → predictions. This is where the
theory from ML-1/ML-2 becomes real code.

## Learning objectives
- LO1 — Load MNIST via torchvision datasets/transforms and build DataLoaders.
- LO2 — Define a small CNN (conv→relu→pool→fc) as an `nn.Module`; explain each layer.
- LO3 — Write the training loop (forward, loss, backward, optimizer step) and understand each line.
- LO4 — Evaluate on the test set (accuracy, confusion matrix) and visualise predictions.

## Scope
In: torchvision MNIST, a small CNN, CPU training loop (few epochs), evaluation, prediction viz.
Out: GPU/large models, data augmentation depth (mention), hyperparameter search.

## Outline
1. What & why — the training loop demystified; the Java analogy for an epoch/batch.
2. Data — MNIST via torchvision; transforms; DataLoader.
3. Model — a small CNN as nn.Module.
4. Train — the loop, loss (CrossEntropy), optimizer (SGD/Adam); loss curve.
5. Evaluate — test accuracy, confusion matrix, sample predictions.
6. Pitfalls — forgetting eval mode / no_grad, wrong loss/shape, not seeding.

## Assets to produce
- Prose: "Machine Learning/Worked Examples/computer-vision/image-classification-mnist.md"
- Code: "Machine Learning/Worked Examples/computer-vision/code/mnist_cnn.py"
- Artefacts: loss/accuracy curve; confusion matrix; a grid of predictions; the trained model file (or note it as regenerable).

## Claims to ground (Haiku, before writing)
- [ ] Verify current torch/torchvision versions + the CPU install command (pytorch.org, 2026); confirm torchvision.datasets.MNIST download works (mirror/licence).
- [ ] Verify the current torchvision transforms API (v2 vs legacy) and nn/optim APIs used.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — mnist_cnn.py TRAINS on CPU in a few minutes and reaches sane accuracy (>97%), producing artefacts; snippet-check passes. Keep epochs small enough to run in-sandbox. AC3 — torch/torchvision versions + APIs grounded. AC4 — the training loop explained line-by-line for a newcomer.

## Gates
Entry: approved; notes landed (esp. torch install). Exit: DoD checklist. NOTE: uses a separate .venv-ml.
