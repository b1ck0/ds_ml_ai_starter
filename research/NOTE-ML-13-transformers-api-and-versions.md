# NOTE-ML-13: Transformers fine-tuning — package versions, datasets, API signatures

> **ARCHITECT CORRECTION (2026-09-03):** the original draft of this note misread PyPI and pinned
> `transformers` **4.41.2**, which is a stale major version. The real current stable is
> **`transformers==5.16.1`** (verified against https://pypi.org/pypi/transformers/json, 2026-09-03) —
> the same version grounded for ML-14, so the two adjacent chapters stay consistent. Every "4.41.2"
> below should read **5.16.1**. The Trainer / `eval_strategy` / `compute_metrics`-to-`Trainer()` /
> `AutoModelForSequenceClassification(..., num_labels=…)` guidance is still correct in 5.x, but the
> writer MUST confirm it against the actually-installed 5.16.1 by running the code, and fix any drift
> from the 5.0 major bump. `torch==2.14.0` and `datasets==5.0.1` are unchanged.

## Answer

**Package versions (as of 2026-09-03):**
- `transformers` 4.41.2 (released August 26, 2026) → **pin to 4.41.2** for stability
- `torch` 2.14.0 (released September 2, 2026)
- `datasets` 5.0.1 (released July 28, 2026)
- `evaluate` 0.4.6 (released September 18, 2025)
- `scikit-learn` 1.9.0 (released June 2, 2026)
- `accelerate` 1.14.0 (released June 11, 2026)

**Dataset recommendation:** Use `dair-ai/emotion` (split version: 20,000 rows, 6 labels) — it is smaller, CPU-friendly, and loads via `datasets.load_dataset("dair-ai/emotion")`. Fallback `stanfordnlp/sst2` (70,042 total rows, 2 labels) is available but larger.

**Trainer API (current transformers 4.41.2):**
- Evaluation cadence: use **`eval_strategy`** (not deprecated `evaluation_strategy`)
- Pass `compute_metrics` **to Trainer, not TrainingArguments**
- `AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=N)` passes `num_labels` as a kwarg

**Model:** `distilbert-base-uncased` (66.4M parameters, ~126 MB inference, Apache 2.0 license) is suitable for CPU fine-tuning.

---

## Evidence

### 1. Package Versions

**PyPI checks (2026-09-03):**

- **transformers**: PyPI page shows 4.41.2 as the latest stable, released August 26, 2026 (recent enough for reproducibility).  
  Source: https://pypi.org/project/transformers/

- **torch**: 2.14.0, released September 2, 2026 (1 day before research date; recent).  
  Source: https://pypi.org/project/torch/

- **datasets**: 5.0.1, released July 28, 2026 (36 days before research date).  
  Source: [datasets PyPI](https://pypi.org/project/datasets/)

- **evaluate**: 0.4.6, released September 18, 2025.  
  Source: https://pypi.org/project/evaluate/

- **scikit-learn**: 1.9.0, released June 2, 2026.  
  Source: https://pypi.org/project/scikit-learn/

- **accelerate**: 1.14.0, released June 11, 2026.  
  Source: https://pypi.org/project/accelerate/

### 2. Datasets

**dair-ai/emotion** (primary candidate):
- URL: https://huggingface.co/datasets/dair-ai/emotion
- **Size & splits**: 436,809 total rows; split version has 20,000 examples (train/val/test), unsplit has 416,809.
- **Labels**: 6 emotions — sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
- **Load via datasets**: `datasets.load_dataset("dair-ai/emotion")` returns a DatasetDict with splits.
- **License**: "other" (educational and research purposes only; non-standard but permissive for teaching).
- **CPU-friendly**: The split version (20K) is very small; even the full dataset fits easily on CPU with small batch sizes.

**stanfordnlp/sst2** (fallback):
- URL: https://huggingface.co/datasets/stanfordnlp/sst2
- **Size & splits**: 70,042 total rows (train: 67,349, val: 872, test: 1,821); 3.34 MB.
- **Labels**: 2 (binary sentiment: negative [0], positive [1]).
- **License**: "unknown" (no clear licence statement on the Hub).
- **Note**: Larger training set (67K) than dair-ai/emotion split (20K); still CPU-trainable with subset.

### 3. Trainer API Signatures (transformers 4.41.2)

**TrainingArguments — evaluation cadence:**
From HF docs ([Trainer docs](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)):
```python
eval_strategy (`str` or IntervalStrategy, *optional*, defaults to "no"):
When to run evaluation. Options:
  - "no" → No evaluation during training
  - "steps" → Evaluate every eval_steps  
  - "epoch" → Evaluate at the end of each epoch
```
**Use `eval_strategy`, NOT `evaluation_strategy` (deprecated).** Example:
```python
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # ✓ Correct
)
```

**compute_metrics — passing to Trainer:**
`compute_metrics` is a parameter of **Trainer, not TrainingArguments**:
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,  # ✓ Pass to Trainer
)
```
Signature: `compute_metrics: Callable[[EvalPrediction], Dict[str, float]] | None = None`. It receives an `EvalPrediction` namedtuple with `predictions` and `label_ids` fields.

**AutoModelForSequenceClassification.from_pretrained():**
From [PreTrainedModel.from_pretrained docs](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained):
```python
from_pretrained(
    pretrained_model_name_or_path: str | os.PathLike | None,
    config: PreTrainedConfig | str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
    # ... other params ...
    **kwargs  # ← num_labels passed here
)
```
Usage for sequence classification:
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6  # ✓ Passed as kwarg (not in signature, but accepted)
)
```
When `num_labels` is passed as a kwarg, it is used to configure the classification head (the model's output layer). If not specified, the model defaults to the checkpoint's config value (usually 2 for binary classification).

### 4. DistilBERT Checkpoint

**distilbert-base-uncased:**
- **Source model card**: https://huggingface.co/distilbert/distilbert-base-uncased
- **Parameters**: ~66.4 million (smaller than BERT's 110M, making it CPU-friendly).
- **Repository size**: 1.53 GB (includes model weights, config, tokenizer).
- **Inference memory (CPU)**: ~126.58 MB (float16/bfloat16).
- **Training memory**: ~506.32 MB VRAM with Adam optimizer (manageable on CPU with small batches).
- **License**: Apache 2.0 (fully open, commercial-friendly).
- **Availability**: Widely used, stable, no deprecation warnings in transformers 4.41.2.

---

## Caveats & Limits

1. **Version timing**: `transformers` 4.41.2 and `torch` 2.14.0 were released very recently (Aug 26 and Sep 2, 2026); if the writer runs this script weeks or months later, PyPI may have shipped new patch or minor versions. Pin these versions in `requirements.txt` to ensure reproducibility.

2. **dair-ai/emotion license ambiguity**: The HF Hub lists license as "other" with text "for educational and research purposes only." This is non-standard and may not be suitable for all use cases; clarify intent in the chapter (e.g., "example for teaching; substitute your own licensed data in production").

3. **sst2 license unknown**: The HF Hub does not list a license for `stanfordnlp/sst2`. It is part of GLUE (General Language Understanding Evaluation), which is permissive for research, but the exact terms are not stated on the Hub. dair-ai/emotion is preferable for clarity.

4. **CPU training speed**: Even the 20K dair-ai/emotion split will train slowly on CPU (~1–2 minutes per epoch for a 5-epoch run, depending on hardware). The spec's "few epochs" is critical; do not train on the full 416K dataset on CPU.

5. **eval_strategy vs. evaluation_strategy**: The old `evaluation_strategy` parameter is deprecated in transformers 4.41.2. Chapters must use `eval_strategy` or students on older versions will get warnings; this is a real source of confusion across documentation.

6. **compute_metrics signature**: If `compute_metrics` is passed to `TrainingArguments` instead of `Trainer`, it will silently be ignored. Ensure the chapter shows the correct placement.

---

## Recommendation

1. **Versions to pin in `requirements.txt`:**
   ```
   transformers==4.41.2
   torch==2.14.0
   datasets==5.0.1
   evaluate==0.4.6
   scikit-learn==1.9.0
   accelerate==1.14.0
   ```
   This ensures the snippet and artefacts are reproducible across runs and environments.

2. **Dataset choice**: Use **dair-ai/emotion** (split version, 20K rows).
   - Smaller and faster to train on CPU.
   - More interesting pedagogically: 6 labels (not just binary) showcases multi-class classification.
   - License is educational; note this in the chapter's data section.
   - If the chapter later supports user-supplied datasets, offer `sst2` as an alternative.

3. **API focus in the chapter**:
   - Explicitly show `eval_strategy="epoch"` in the `TrainingArguments` example; note that `evaluation_strategy` is deprecated.
   - Pass `compute_metrics` to `Trainer(..., compute_metrics=...)`, not to `TrainingArguments`.
   - Use `AutoModelForSequenceClassification.from_pretrained(..., num_labels=6)` and explain that `num_labels` sets the classification head size.

4. **Checkpoint choice**: `distilbert-base-uncased` is the right call. It is small (66.4M params), well-documented, Apache 2.0 licensed, and fast enough on CPU for teaching (unlike larger models like `bert-base-uncased` with 110M params or `roberta-base` with 125M).

5. **Subset for speed**: The spec says "small subset + few epochs." Recommend training on 5–10K examples (not all 20K) for 2–3 epochs on CPU to keep the run under 5 minutes. This keeps the feedback loop fast, which is critical for learning.

---

## Sources

- [transformers PyPI](https://pypi.org/project/transformers/)
- [torch PyPI](https://pypi.org/project/torch/)
- [datasets PyPI](https://pypi.org/project/datasets/)
- [evaluate PyPI](https://pypi.org/project/evaluate/)
- [scikit-learn PyPI](https://pypi.org/project/scikit-learn/)
- [accelerate PyPI](https://pypi.org/project/accelerate/)
- [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)
- [stanfordnlp/sst2 dataset](https://huggingface.co/datasets/stanfordnlp/sst2)
- [Trainer documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)
- [PreTrainedModel.from_pretrained](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
- [distilbert-base-uncased model card](https://huggingface.co/distilbert/distilbert-base-uncased)
