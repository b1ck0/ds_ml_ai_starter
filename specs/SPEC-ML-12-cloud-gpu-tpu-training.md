# SPEC-ML-12: Cloud — GPU/TPU training and blob storage (Google/AWS/Azure)

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Machine Learning
**Section:** Cloud Environment Setup
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-0, SPEC-ML-4
**Nature:** GROUNDED CONCEPTUAL — no cloud execution in the sandbox. Real, verified CLI/SDK snippets as
reference; no fabricated output.

## Intent
When a model outgrows the laptop, training moves to cloud accelerators with data in blob storage.
Map the story across the three clouds so a Java dev recognises the equivalent service, and explain the
GPU-vs-TPU choice and the data-loading realities.

## Learning objectives
- LO1 — Explain when you need a GPU/TPU (big models, big data) and the cost/time trade-off vs CPU.
- LO2 — Map blob storage + accelerator training across clouds: GCS + Vertex/GKE + TPU/GPU; S3 + SageMaker/EC2 + GPU; Azure Blob + Azure ML + GPU.
- LO3 — Read a minimal "train on a cloud GPU with data from blob storage" job definition in one cloud.
- LO4 — Understand GPU vs TPU (when each helps) and efficient data loading (sharding, streaming datasets).

## Scope
In: cross-cloud mapping table; GPU vs TPU; blob storage for datasets/checkpoints; one worked (reference) cloud training-job sketch; cost/quota realities.
Out: hands-on execution; distributed-training internals depth (mention + link).

## Outline
1. Why leave the laptop — the compute/data wall; accelerators.
2. GPU vs TPU — architectures, when each wins, availability.
3. Cross-cloud mapping TABLE — storage + training service + accelerator per cloud.
4. A reference training job — e.g. a Vertex AI / SageMaker training job reading from blob storage, saving checkpoints back.
5. Data loading at scale — sharded/streaming datasets; the I/O bottleneck.
6. Pitfalls — accelerator quotas, egress cost, tiny-batch underutilisation, forgetting to checkpoint.

## Claims to ground (Haiku, before writing) — VERIFY 2026 service names
- [ ] Verify current service names + the mapping: Google (Cloud Storage, Vertex AI custom training / GKE, Cloud TPU, GPUs), AWS (S3, SageMaker training / EC2, Trainium/Inferentia, GPUs), Azure (Blob Storage, Azure ML, GPUs). Confirm against current official docs.
- [ ] Verify the reference training-job snippet's SDK/CLI for ONE cloud (e.g. Vertex AI CustomJob or SageMaker Estimator) from official docs — mark clearly as reference, not executed.

## Assets to produce
- Prose: "Machine Learning/Cloud Environment Setup/gpu-tpu-training-and-storage.md"
- Artefacts: the cross-cloud comparison table (in-prose) + a training-workflow diagram (matplotlib/SVG).

## Acceptance criteria
- [ ] AC1 — LOs delivered incl. the mapping table + GPU/TPU decision. AC2 — every code block is either runnable (the diagram) or clearly fenced reference; snippet-check passes; NO fabricated console output. AC3 — service names + the reference SDK snippet grounded with dated citations. AC4 — the "when do I actually need this" question answered honestly for a cost-conscious engineer.

## Gates
Entry: approved; grounding (2026 service names) landed. Exit: DoD checklist.
