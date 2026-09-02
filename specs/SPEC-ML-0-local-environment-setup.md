# SPEC-ML-0: Local Environment Setup (Machine Learning)

**Status:** approved
**Subject:** Machine Learning
**Section:** Local Environment Setup
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-0 (Python basics)

## Intent
Stand up the deep-learning toolchain and explain the CPU/GPU/CUDA story a Java dev hasn't met before.
This env underpins every ML chapter; keep it CPU-runnable so the examples work without a GPU.

## Learning objectives
- LO1 — Install PyTorch + torchvision (CPU build) and TensorFlow; verify each imports and reports its version + device.
- LO2 — Explain tensors vs numpy arrays, and the CPU/GPU/CUDA/MPS device model; check `torch.cuda.is_available()`.
- LO3 — Understand why a separate venv (`.venv-ml`) is used and why versions must match the CUDA/CPU build.
- LO4 — Know when you actually need a GPU (training big nets) vs when CPU is fine (these chapters).

## Scope
In: PyTorch/torchvision/TensorFlow install (CPU), device checks, tensor basics, the venv rationale.
Out: CUDA driver setup depth (link to official), distributed training (→ ML cloud chapter).

## Outline
1. What & why — the DL stack vs the DS stack; why a fresh venv.
2. Install PyTorch/torchvision (CPU wheel) + TensorFlow; the version/index-url gotcha.
3. Verify — a script printing versions + device availability.
4. Tensors 101 — array-like, but with autograd and a device; contrast with numpy.
5. Pitfalls — mismatched CUDA/torch builds, giant downloads, mixing envs.

## Assets to produce
- Prose: "Machine Learning/Local Environment Setup/local-environment-setup.md"
- Code: "Machine Learning/Local Environment Setup/code/verify_ml_env.py"
- Artefacts: captured version/device output (```text).

## Claims to ground (Haiku, before writing)
- [ ] Verify current PyTorch + torchvision + TensorFlow versions on PyPI and the CORRECT CPU install command / index URL from pytorch.org (2026).
- [ ] Confirm torch device-check APIs (torch.cuda.is_available, torch.backends.mps).

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — verify_ml_env.py runs on CPU + prints versions/device; snippet-check passes. AC3 — install commands + versions grounded. AC4 — tensor-vs-array + device model explained for a newcomer.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.
