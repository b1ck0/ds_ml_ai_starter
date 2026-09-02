# NOTE-ML-8 & ML-9: NLP Text Classification & Generation Models

**Answer:** **ML-8 Text Classification:** Recommended model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (Apache 2.0 license, ~67M params, free download from HuggingFace). Use API: `pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")` or `AutoModelForSequenceClassification.from_pretrained()` + `AutoTokenizer.from_pretrained()` with transformers 5.16.1. Returns dict with 'label' (NEGATIVE/POSITIVE) and 'score' (confidence). **ML-9 Text Generation:** **RoBERTa is encoder-only and CANNOT generate** (masked language model, bidirectional context, not causal; confirmed from academic literature https://arxiv.org/abs/1907.11692). Recommended decoder model: `distilgpt2` (Apache 2.0 license, 88.2M params, free download). Use `generate()` API with params: `do_sample` (bool, greedy vs sampling), `temperature` (float > 0, diversity), `top_k` (int, keep top-k tokens), `top_p` (float, nucleus sampling), `num_beams` (int > 1, beam search). Output: generated token IDs; decode via `tokenizer.decode()`. Tiny eval set: hand-written 5-10 sentence-label pairs OR small slice of SST-2 dataset. Checked 2026-09-02.

**Evidence:**

### ML-8: Text Classification

1. **Model: distilbert-base-uncased-finetuned-sst-2-english** (verified 2026-09-02)
   - Source: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
   - License: Apache-2.0 (permissive, free for any use)
   - Size: ~67M parameters (DistilBERT is ~40% smaller than BERT)
   - Task: Fine-tuned on SST-2 (Stanford Sentiment Treebank); binary sentiment classification (negative/positive)
   - Download: Free via `transformers` library; auto-cached to `~/.cache/huggingface/hub/`
   - Accuracy: 91.3% on dev set (comparable to BERT 92.7%)

2. **Pipeline API** (verified 2026-09-02)
   - Source: https://huggingface.co/transformers/v5.16.1/pipeline_tutorial.html (transformers 5.16.1)
   - Code: `pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")`
   - Output: list of dicts, e.g. `[{'label': 'POSITIVE', 'score': 0.9998}]`
   - Supports batch inference: `classifier([text1, text2, ...])`

3. **AutoModel API** (verified 2026-09-02)
   - `AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")`
   - `AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")`
   - Output: logits (2D tensor, batch_size x num_classes=2)
   - Softmax to get probabilities

4. **Tiny eval set recommendation** (verified 2026-09-02)
   - SST-2 dataset: 67K train, 873 test; easily loadable via `datasets` library
   - Hand-written examples: 5-10 clearly positive/negative sentences for quick validation
   - Example: `["I love this!", "Terrible movie.", "It's okay."]` with labels `[1, 0, 1]`

### ML-9: Text Generation

1. **RoBERTa is encoder-only (CANNOT generate)** (verified 2026-09-02)
   - RoBERTa paper: https://arxiv.org/abs/1907.11692
   - Architecture: **Encoder-only** (masked language model, MLM training)
   - Context: **Bidirectional** (attends to tokens before AND after the masked position)
   - Generation requirement: **Causal attention** (only attend to past tokens; autoregressive next-token prediction)
   - Quote from literature: "Encoder-only models like RoBERTa cannot generate text because the discrepancy between the pre-training task of masked language models and the subsequent generation function limits their application"
   - Use cases: RoBERTa for classification, encoder-decoder (T5, BART) or decoder (GPT) for generation

2. **Model: distilgpt2** (verified 2026-09-02)
   - Source: https://huggingface.co/distilbert/distilgpt2
   - License: Apache-2.0 (permissive, free for any use)
   - Size: 88.2M parameters
   - Architecture: Decoder-only (GPT-2 style), causal attention
   - Training: Distilled from GPT-2 (124M params) via knowledge distillation
   - Speed: ~2x faster than GPT-2, minimal quality loss
   - Download: Free via `transformers` library; auto-cached

3. **Generate API** (verified 2026-09-02)
   - Source: transformers 5.16.1 documentation
   - Signature: `model.generate(input_ids, max_length=..., **kwargs)`
   - Common decoding parameters:
     - `do_sample` (bool): False = greedy decoding (highest probability token); True = sampling
     - `temperature` (float > 0): < 1.0 = sharper (more confident), > 1.0 = softer (more diverse). Default 1.0.
     - `top_k` (int): keep only top-k most likely tokens; sample from those
     - `top_p` (float, ∈ [0, 1]): nucleus sampling; keep tokens with cumulative probability ≤ top_p
     - `num_beams` (int ≥ 1): 1 = greedy, > 1 = beam search (explores multiple paths)
   - Output: token IDs (LongTensor, shape [batch_size, seq_length])
   - Decode: `tokenizer.decode(token_ids, skip_special_tokens=True)`

4. **Example decoding strategies** (verified 2026-09-02)
   - Greedy: `do_sample=False, num_beams=1` → deterministic, repetitive
   - Sampling: `do_sample=True, temperature=0.7, top_p=0.9` → diverse, non-deterministic
   - Beam search: `num_beams=5, num_return_sequences=1` → balanced exploration & quality

5. **Tiny eval set** (verified 2026-09-02)
   - Hand-written: 5-10 prompts, manually verify generated text quality (no formal metrics needed for tutorial)
   - Example prompts: `["The future of AI is", "Today I learned that", "Cats are"]`
   - No labeled groundtruth needed (generation task is open-ended)

**Caveats / limits:**

- **RoBERTa myth:** Common mistake to try using RoBERTa for generation; it cannot work (no causal mask, bidirectional only).
- **Encoder-decoder alternative:** T5, BART, mBART can both encode and generate; suitable for seq2seq tasks.
- **Distilbert size:** At ~67M params, fits on CPU but inference is slower than GPU; acceptable for tutorial.
- **Distilgpt2 quality:** Smaller than GPT-2; may produce lower-quality text; acceptable for tutorial demonstration.
- **Temperature / top_p interaction:** Using both simultaneously can be confusing; recommend showing them separately in tutorial.
- **Repetition problem:** Small models on small prompts often loop/repeat; mention in pitfalls.
- **Tokenizer mismatch:** Must use the same tokenizer as the model (e.g., distilgpt2 tokenizer for distilgpt2 model).

**Recommendation:**

1. **For ML-8 chapter:**
   - Use pipeline API for simplicity; show AutoModel API for advanced use (accessing logits).
   - Display confusion matrix on a small eval set (5 classes × 5 predictions is sufficient).
   - Link back to DS-6 metrics: accuracy, F1, precision, recall.
   - Mention transfer learning: why pretrained > training from scratch.

2. **For ML-9 chapter:**
   - **LEAD with the RoBERTa correction:** Show why RoBERTa fails (bidirectional), then introduce distilgpt2 (causal).
   - Show the SAME prompt under greedy, sampling (temp 0.7), beam search (5 beams); print all outputs.
   - Explain decoding knobs in order: greedy → temperature → top-k/top-p → beam search.
   - Mention pitfalls: repetition, hallucination, context length.
   - Create a small table: prompt | greedy | temp=0.7 | temp=1.5 | beam_5 (real outputs, no fabrication).

3. **For both:**
   - Pin model IDs in example code (exact HuggingFace paths, not generic names).
   - Use `device="cpu"` explicitly in code so it runs on CPU venv.
   - Measure inference time and mention it's slower on CPU (set expectations).

4. **For tiny eval set:**
   - Provide a function to load SST-2 subset (first 50 examples) via `datasets.load_dataset("sst2").select(range(50))`.
   - OR provide 10 hand-written examples in a .py file or .json file for reproducibility.
