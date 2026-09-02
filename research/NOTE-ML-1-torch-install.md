# NOTE-ML-1: PyTorch/TensorFlow CPU Installation & Device APIs

**Answer:** PyTorch 2.14.0 + torchvision 0.29.0 (released 2026-09-02); install via `pip install torch==2.14.0 torchvision==0.29.0 torchaudio==2.14.0 --index-url https://download.pytorch.org/whl/cpu`; TensorFlow 2.21.0 (released 2026-03-06); device check via `torch.cuda.is_available()` and `torch.backends.mps.is_available()`; torchvision.datasets.MNIST mirrors at S3 (ossci-datasets) and Yann LeCun's server; transforms v2 API recommended.

**Evidence:**

1. **PyTorch latest version:** 2.14.0 released 2026-09-02 (checked 2026-09-02)
   - Source: https://pypi.org/project/torch/
   - "Tensors and Dynamic neural networks in Python with strong GPU acceleration"

2. **Torchvision latest version:** 0.29.0 released 2026-09-02 (checked 2026-09-02)
   - Source: https://pypi.org/project/torchvision/
   - 24 files available for Python 3.10-3.14 across Windows, Linux, macOS

3. **CPU install index-url pattern** (verified 2026-09-02)
   - From PyTorch forums & official docs: `--index-url https://download.pytorch.org/whl/cpu`
   - Sources: https://pytorch.org/get-started/locally/ and https://discuss.pytorch.org/t/index-url-to-install-pytorch/198253

4. **TensorFlow latest version:** 2.21.0 released 2026-03-06 (checked 2026-09-02)
   - Source: https://pypi.org/project/tensorflow/
   - CPU-compatible; supports Python 3.10-3.13; `pip install tensorflow` installs CPU build by default

5. **torch device-check APIs:**
   - `torch.cuda.is_available()` → returns bool; checks for CUDA capability
   - `torch.backends.mps.is_available()` → returns bool; checks for Metal Performance Shaders (Apple Silicon)
   - Sources: https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html and PyTorch forums

6. **MNIST dataset in torchvision:**
   - Class: `torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)`
   - Download mirrors: `https://ossci-datasets.s3.amazonaws.com/mnist/` and `http://yann.lecun.com/exdb/mnist/`
   - Known issues: occasional 403 errors on Yann LeCun's server; S3 mirror more reliable
   - Source: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html and GitHub issue #8568

7. **Torchvision transforms v2 API:**
   - Recommended API; supports not just images but bounding boxes, masks, videos, keypoints
   - Import: `from torchvision import transforms as tv_transforms; tv_transforms.v2`
   - Faster than legacy v1, full backward compatible
   - Source: https://docs.pytorch.org/vision/stable/transforms.html
   - Quote: "new features and improvements will only be considered for the v2 transforms"

**Caveats / limits:**

- **CPU download size:** ~250 MB for torch CPU wheel; ~700 MB for torchvision. Not a CUDA build (~2 GB+) so significantly smaller on CPU-only machines.
- **Version moving:** Versions checked on 2026-09-02; newer releases may be available. Pin versions in requirements.txt.
- **MNIST mirror reliability:** Yann LeCun's server (http://yann.lecun.com/exdb/mnist/) has experienced intermittent 403 errors. S3 mirror (ossci-datasets) is more stable; consider fallback or pre-download.
- **MPS vs CUDA:** `torch.backends.mps.is_available()` requires macOS 12.3+ with Apple Silicon; not relevant on Windows/Linux.
- **TensorFlow vs PyTorch:** Both installable on same venv but no requirement in spec to mix them; recommend separate sections or clear separation in env setup.

**Recommendation:**

1. **Pin the exact versions** in requirements.txt:
   ```
   torch==2.14.0
   torchvision==0.29.0
   torchaudio==2.14.0
   tensorflow==2.21.0
   ```

2. **Use the CPU index-url in pip install:**
   ```bash
   pip install torch==2.14.0 torchvision==0.29.0 torchaudio==2.14.0 --index-url https://download.pytorch.org/whl/cpu
   pip install tensorflow==2.21.0
   ```

3. **Device-check script should use:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

4. **For MNIST download reliability:** Ensure the demo code handles mirror fallback or pre-caches the dataset.

5. **Use torchvision.transforms.v2 for all new transforms code** — no deprecation warning, forward-compatible.
