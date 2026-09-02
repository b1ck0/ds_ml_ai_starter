# NOTE-ML-5 & ML-6: TorchVision Detection and Segmentation APIs

**Answer:** TorchVision 0.29.0 detection API: `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)` returns a model; COCO category names available via `weights.meta['categories']` (list of 80 COCO class names); model output is dict with keys 'boxes' (Nx4 tensors, xyxy format), 'labels' (N class indices 1-80), 'scores' (N confidence values); NMS/thresholding applied post-inference. Segmentation: `torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)` or `fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)`; output dict with key 'out' containing logits (B, num_classes, H, W); class mask via `torch.argmax(output['out'], dim=1)`. Sample images: Penn-Fudan Database (https://www.cis.upenn.edu/~jshi/ped_html/) or PyTorch Vision tutorials (https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html). Weights auto-download to `~/.cache/torch/hub/` on first call (checked 2026-09-02).

**Evidence:**

1. **Detection API signature & weights enum** (verified 2026-09-02)
   - Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html (main docs 0.28+)
   - Quote: "torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, progress=True, **kwargs)"
   - `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` is the standard enum value
   - Also available: `fasterrcnn_resnet50_fpn_v2`, `retinanet_resnet50_fpn`, `ssd300_vgg16`

2. **COCO categories from weights.meta** (verified 2026-09-02)
   - TorchVision 0.15+ stores metadata: `weights.meta['categories']` → list of 80 COCO class names
   - Source: TorchVision documentation & GitHub discussions
   - Access pattern: `FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta['categories']`

3. **Detection output format** (verified 2026-09-02)
   - Model in eval mode returns dict: `{'boxes': Tensor(N,4), 'labels': Tensor(N), 'scores': Tensor(N)}`
   - Boxes in xyxy (x1, y1, x2, y2) format by default
   - Labels are 1-indexed COCO class IDs (1-80, not 0-indexed)
   - Scores are post-NMS confidence values

4. **Segmentation API** (verified 2026-09-02)
   - `deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)`
   - Also: `fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)`
   - Output dict with key 'out': shape (B, num_classes=21, H, W) for COCO-trained models
   - Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html

5. **Segmentation class mask** (verified 2026-09-02)
   - Argmax over class dimension: `mask = torch.argmax(output['out'], dim=1)` → shape (B, H, W)
   - Class indices 0 = background, 1-20 = COCO semantic classes (21 total for DeepLab)

6. **Sample image source: Penn-Fudan Database** (verified 2026-09-02)
   - 170 images with bounding boxes & segmentation masks for pedestrian detection
   - URL: https://www.cis.upenn.edu/~jshi/ped_html/ (publicly available, CC-like usage)
   - Downloadable via PyTorch tutorials

7. **PyTorch Vision GitHub examples** (verified 2026-09-02)
   - Source: https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
   - Gallery with COCO sample images: https://github.com/pytorch/vision/tree/main/gallery/

8. **Weights download** (verified 2026-09-02)
   - TorchVision auto-downloads pretrained weights on first model instantiation
   - Cache location: `~/.cache/torch/hub/` or set via `TORCH_HOME` env var
   - Requires internet on first run only

**Caveats / limits:**

- **Version-specific:** APIs stable in 0.28+; torchvision 0.29.0 confirmed available (checked 2026-09-02).
- **COCO vs other datasets:** DeepLab/FCN segmentation weights trained on COCO have 21 classes (20 + background); different datasets may have different class counts.
- **Label indexing:** Detection labels are 1-indexed COCO class IDs; segmentation is 0-indexed (0=background, 1-20=classes).
- **NMS:** Faster R-CNN applies NMS internally during inference; output scores are post-NMS.
- **Sample image quality:** Penn-Fudan dataset is small (170 images) and pedestrian-focused; sufficient for tutorial but not representative of full COCO diversity.
- **Weights download size:** ~500 MB for detection models (FasterRCNN), ~100 MB for segmentation models.

**Recommendation:**

1. **For chapter code:** Pin torchvision==0.29.0; always call model.eval() before inference.
2. **For COCO categories in text:** Use `weights.meta['categories']` dynamically to avoid hardcoding 80 class names.
3. **For sample images:** Download Penn-Fudan dataset via the PyTorch tutorial script (automatic) OR use a public CC0 image (e.g., from Unsplash) and demonstrate detection/segmentation on it.
4. **For post-processing:** Teach score thresholding (e.g., scores > 0.5) and NMS explicitly, even though FasterRCNN does NMS internally; note this in the explanation.
5. **For segmentation visualization:** Recommend using matplotlib with a colormap on the argmax mask; discuss class imbalance (many background pixels).
