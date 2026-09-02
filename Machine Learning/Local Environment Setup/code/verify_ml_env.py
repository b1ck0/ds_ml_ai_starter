"""Verify the ML local environment.

Prints the installed versions of PyTorch, torchvision, NumPy, and (if
present) TensorFlow, reports which compute devices torch can see
(CPU / CUDA / MPS), and runs a tiny tensor-vs-numpy-array contrast that
shows the two things a numpy array does not have: a `device` and autograd.

Run this with the ML virtualenv's interpreter, NOT the default DS one --
this project keeps a separate venv for the deep-learning stack (see the
chapter this script belongs to for why):

    .venv-ml\\Scripts\\python.exe "Machine Learning/Local Environment Setup/code/verify_ml_env.py"   # Windows
    .venv-ml/bin/python "Machine Learning/Local Environment Setup/code/verify_ml_env.py"              # macOS/Linux

Expects (pinned in this chapter, see local-environment-setup.md):
    torch==2.14.0 torchvision==0.29.0 numpy==2.5.2
TensorFlow is optional for this chapter -- the script degrades gracefully
if it is not installed in the active venv.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision


def print_versions() -> None:
    print(f"torch version:       {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"numpy version:       {np.__version__}")

    try:
        import tensorflow as tf  # optional -- not required by this chapter's ACs
    except ImportError:
        print("tensorflow:          not installed in this venv (optional here -- see chapter)")
    else:
        print(f"tensorflow version:  {tf.__version__}")


def print_devices() -> None:
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    print(f"torch.cuda.is_available():         {cuda_available}")
    print(f"torch.backends.mps.is_available(): {mps_available}")

    if cuda_available:
        print(f"  CUDA device 0: {torch.cuda.get_device_name(0)}")
    elif mps_available:
        print("  Apple Silicon GPU (MPS) available.")
    else:
        print("  No GPU backend detected on this machine -- tensors default to the")
        print("  CPU device, which is all the examples in these chapters need.")


def tensor_vs_numpy_contrast() -> None:
    print("\n--- numpy array vs. torch tensor ---")

    # A numpy array: a block of memory, always on the CPU, no notion of
    # "device" and no memory of the operations that produced it.
    np_arr = np.array([1.0, 2.0, 3.0])
    print(f"numpy array:  {np_arr}  dtype={np_arr.dtype}")

    # A torch tensor looks the same on the surface, but carries two things
    # numpy arrays never do: a `device` (which piece of hardware the data
    # lives on) and, with requires_grad=True, an autograd tape that records
    # every operation so it can be differentiated later.
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print(f"torch tensor: {t}  dtype={t.dtype}  device={t.device}  requires_grad={t.requires_grad}")

    # autograd in action: torch can differentiate y = sum(t**2) with respect
    # to t automatically. numpy has no equivalent -- you would have to work
    # out and code the derivative (2t) by hand.
    y = (t**2).sum()
    y.backward()
    print(f"y = sum(t**2)     = {y.item()}")
    print(f"dy/dt (t.grad)    = {t.grad}  (expected: 2*t = [2., 4., 6.])")


def main() -> None:
    print("=== Versions ===")
    print_versions()
    print("\n=== Devices ===")
    print_devices()
    tensor_vs_numpy_contrast()


if __name__ == "__main__":
    main()
