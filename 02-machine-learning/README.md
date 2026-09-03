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
  (distilgpt2, a decoder — RoBERTa is encoder-only and cannot generate, explained in the generation chapter);
  fine-tuning a transformer (DistilBERT trained end to end on dair-ai/emotion — the explicit PyTorch
  loop, then the HF `Trainer`, save/reload, inference); text & NLP metrics (classification — macro/micro/
  weighted F1; generation — perplexity, BLEU, chrF, ROUGE, BERTScore; retrieval/similarity — cosine
  similarity, Recall@k, MRR, nDCG — each computed by hand and reproduced with a pinned library).
- **LLMs** — the transformer from the inside; text generation.
- **Reinforcement Learning** — the MDP, policy/value/Bellman, ε-greedy, Q-learning (off-policy) vs.
  SARSA (on-policy); a tabular Q-learning agent that learns to hunt down a Rook on a tiny, genuinely-chess
  corner-capture task, evaluated against random and greedy baselines; DQN, policy gradients, and
  self-play + MCTS (AlphaZero) explained and grounded (not executed), with the honest compute-gap
  caveat.

## Cloud Environment Setup
Google / AWS / Azure — blob storage · GPU training · TPU training.

## Production Considerations
Serving, quantization, and cost of GPU/TPU training — cross-referenced from the Data Science
production material where it overlaps.
