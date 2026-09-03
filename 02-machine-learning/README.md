# Machine Learning

Part 2 of the crash course. See [`../docs/curriculum.md`](../docs/curriculum.md) for the full
backlog. Chapters are written from approved `specs/SPEC-ML-*.md`.

## Theory
Neural networks · gradient descent · neurons · activation functions · dense / dropout / convolution /
LSTM-GRU layers · transformer · quantized models · fine-tuning · encoder–decoder · autoencoder ·
tokenizers · word2vec · cosine similarity · Euclidean distance.

## Local Environment Setup
Python · TensorFlow · PyTorch · torchvision.

## Worked Examples
- **Computer Vision** — image classification (MNIST), object detection (COCO), semantic segmentation
  (COCO); metrics: mAP, mAR, IoU.
- **Natural Language** — text classification (DistilBERT, a small pretrained encoder); text generation
  (distilgpt2, a decoder — RoBERTa is encoder-only and cannot generate, explained in the generation chapter).
- **LLMs** — the transformer from the inside; text generation.

## Cloud Environment Setup
Google / AWS / Azure — blob storage · GPU training · TPU training.

## Production Considerations
Serving, quantization, and cost of GPU/TPU training — cross-referenced from the Data Science
production material where it overlaps.
