# Cloud GPU/TPU training and blob storage — Google, AWS, Azure

*Machine Learning · Cloud Environment Setup · SPEC-ML-12*

**Nature of this chapter: grounded conceptual.** No cloud account exists in this sandbox, so every
CLI/SDK block below is **reference only** — real, current API surface pulled from each vendor's own
documentation and verified live today, but not executed. Nothing in this chapter is a fabricated
console output: where you'd normally see a job status or a printed number, the text cites the
documented behaviour instead of inventing one. The one piece of code that actually runs is the
workflow diagram generator in `code/`.

## 1. What & why — the compute/data wall

Every model you've trained so far in this curriculum ran on the CPU build of PyTorch/TensorFlow on
your own machine ([NOTE-ML-1](../../research/NOTE-ML-1-torch-install.md):
`pip install torch==2.14.0 torchvision==0.29.0 torchaudio==2.14.0 --index-url
https://download.pytorch.org/whl/cpu`). That's not a toy setup — MNIST-scale image classification,
a small transformer, a text classifier fine-tune all finish in minutes on a laptop CPU. Keep using
it as long as it works; it's free and it's simpler.

Two things push you off the laptop:

- **The model doesn't fit the time budget.** A CPU can do the same floating-point matrix multiplies
  a GPU/TPU does — just roughly one to two orders of magnitude slower per step, because a CPU has a
  handful of general-purpose cores where an accelerator has thousands of cores purpose-built for
  exactly the dense matrix-multiply the forward/backward pass is made of. A fine-tune that takes 20
  minutes on a laptop for a toy example takes days on the same laptop for a real model with tens of
  millions of parameters and a real dataset.
- **The data doesn't fit the machine.** Once a training set is bigger than local disk (image/video
  corpora, large text corpora), there's no "download it first" option — it has to stream from
  somewhere durable and shared. That "somewhere" is cloud object storage, whether or not you also
  need a faster processor.

This is the same trade-off a Java engineer already knows from switching a batch job from a laptop
script to a provisioned cluster: you pay for managed compute and you accept operational overhead
(quotas, IAM, network egress) in exchange for a job finishing in hours instead of weeks, or fitting
in memory at all. It is **not** a trade you make by default — Section 5 comes back to exactly when
it's worth it.

## 2. GPU vs TPU — when each wins

Both are accelerators built to do one thing fast: dense matrix multiplication, the operation that
dominates a neural network's forward and backward pass. They get there differently, and the
difference has practical consequences for which one to reach for.

A **GPU** (NVIDIA T4/L4/A100/H100/H200, across all three clouds) is a general-purpose massively
parallel processor: thousands of smaller cores, a mature and very wide software ecosystem (CUDA,
cuDNN, every ML framework), and the ability to run arbitrary custom kernels. A **TPU** (Google's
Tensor Processing Unit) is Google's purpose-built ASIC for ML: its core is a **128×128 systolic
array** called the Matrix Multiply Unit (MXU), paired with high-bandwidth on-chip memory, built
specifically to execute the giant matrix operations training is made of
[source: Cloud TPU introduction](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) (checked
2026-09-02).

Google's own guidance on when to pick which (same source, checked 2026-09-02) is a good working
rule:

- **Reach for a TPU when:** the model is "dominated by matrix computations," it's a large model
  trained with a large effective batch size, or it's going to "train for weeks or months" — the
  MXU's efficiency compounds over a long run.
- **Reach for a GPU when:** the model has "a significant number of custom PyTorch/JAX operations,"
  uses TensorFlow ops that aren't available on Cloud TPU, or is a medium-to-large model where you
  need the flexibility of a mature, general-purpose kernel ecosystem rather than an XLA-compiled,
  matrix-multiply-shaped graph.

For a Java engineer's mental model: a GPU is the JVM of accelerators — general-purpose, huge
ecosystem, runs almost anything you throw at it, at a small efficiency cost. A TPU is closer to a
purpose-built appliance — extremely efficient at exactly the workload it was designed for, less
forgiving outside it. TPUs are also **Google-only**; AWS and Azure answer the "purpose-built
silicon" question differently (Section 3) rather than shipping a TPU equivalent.

Practically, for the worked examples earlier in this curriculum (MNIST classification, a text
classifier fine-tune, small transformer experiments), you're squarely in "GPU, and probably don't
even need a TPU" territory — those models aren't dominated by the kind of massive, sustained
matrix-multiply workload where a TPU's advantage shows up. TPUs earn their keep on much larger,
longer-running training runs.

## 3. Cross-cloud mapping — storage, training service, accelerators

The same three ingredients — a place to put the data, a managed service to run the training job,
and the accelerator hardware itself — exist on every cloud under different names. All verified live
against each vendor's current documentation (checked 2026-09-02):

| | **Google Cloud** | **AWS** | **Azure** |
|---|---|---|---|
| **Blob/object storage** | Cloud Storage (GCS) | S3 | Blob Storage |
| **Managed training service** | Vertex AI custom training jobs — `CustomTrainingJob` / `CustomJob` in the `google-cloud-aiplatform` SDK, pinned `==2.1.0` ([NOTE-18](../../research/NOTE-18-managed-platforms.md); [source: PyPI](https://pypi.org/project/google-cloud-aiplatform/), checked 2026-09-02) | SageMaker Training — the unified `ModelTrainer` class (SDK v3), which now replaces the older per-framework `Estimator` classes (`PyTorch`, `TensorFlow`, `HuggingFace`, `XGBoost`, …) [source: SageMaker checkpointing guide](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints-enable.html) (checked 2026-09-02) | Azure ML jobs — `command()` submitted via `MLClient` (Python SDK v2, `azure-ai-ml`) [source: Train ML models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-model?view=azureml-api-2) (checked 2026-09-02) |
| **GPU accelerators** | NVIDIA T4, L4, A100 (40/80GB), H100 80GB, H200 141GB — set via `accelerator_type` / `accelerator_count` on the worker pool spec [source: Vertex AI compute config](https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute) (checked 2026-09-02) | EC2 P4d (8×A100), P5 (8×H100 80GB, NVSwitch), P5e/P5en (H200) [source: EC2 accelerated computing](https://aws.amazon.com/ec2/instance-types/accelerated-computing/); [P5 in SageMaker](https://aws.amazon.com/about-aws/whats-new/2025/08/p5-instance-nvidia-h100-gpu-sagemaker-training-processing-jobs/) (checked 2026-09-02) | NC-series (H100 NVL, single-GPU/no-InfiniBand tier) and ND-series (H100 SXM5 with NVLink + InfiniBand, up to 8 GPUs/VM; H200 in preview) [source: NC-family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family), [ND-family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-family) (checked 2026-09-02) |
| **Custom ML silicon (non-GPU)** | Cloud TPU v5e / v6e — `ct5lp-hightpu-{1,4,8}t` machine types, `tpuTopology`; requires JAX ≥0.4.6, TensorFlow ≥2.15, or PyTorch ≥2.1 [source: Training with TPU accelerators](https://docs.cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm) (checked 2026-09-02) | Trainium (Trn1/Trn2, training-optimised) and Inferentia (Inf1/Inf2, inference-optimised) — usable from SageMaker with the same `ModelTrainer`/estimator surface, e.g. `ml.trn1.32xlarge` [source: EC2 accelerated computing](https://aws.amazon.com/ec2/instance-types/accelerated-computing/) (checked 2026-09-02) | none — Azure's accelerator story is GPU-only; no Azure-specific training ASIC ships in Azure ML today (checked 2026-09-02, absence confirmed against the same Azure VM sizing docs above) |

**A naming note carried over from the sibling Data Science chapter:** Google rebranded Vertex AI to
"Gemini Enterprise Agent Platform" in 2026, but the underlying services, SDK, and API surface named
above are unchanged — see
[Data Science/Cloud Environment Setup/managed-ml-platforms.md §3](../../Data%20Science/Cloud%20Environment%20Setup/managed-ml-platforms.md)
for the citation trail. This chapter keeps calling it "Vertex AI" because that's the name the SDK,
its docs, and its PyPI package still use.

Two practical read-throughs of the table:

- **TPUs are the one row that doesn't have a direct AWS/Azure equivalent.** Trainium is AWS's
  answer to "purpose-built training silicon," but it's a different architecture (not a systolic
  array/MXU design) with its own compiler toolchain — it's not a drop-in TPU. If your workload is
  chosen specifically *for* the TPU's MXU, it's a Google Cloud–only decision.
- **The blob storage row is the most portable one.** GCS, S3, and Blob Storage all speak roughly the
  same "bucket of objects, addressed by key, read as a byte stream" model. Data-loading code that
  streams shards (Section 5) ports across clouds far more easily than the training-job submission
  code does.

## 4. A reference training job — Vertex AI, reading from GCS, checkpointing back

LO3. Every cloud's training job answers the same question — "run this container/script on N
accelerators, give it these inputs, keep what it writes" — with a different SDK. Below is the
**Vertex AI** shape end to end (verified against the SDK's own PyPI page and its compute-config
docs, both checked 2026-09-02), followed by the equivalent shape on AWS and Azure so you can
recognise it wherever you land.

The dataset lives in a GCS bucket, sharded (Section 5 explains why); the job streams it in, trains,
and writes checkpoints to a second GCS prefix as it goes — the exact loop Figure 1 draws.

![Cloud GPU/TPU training workflow: object/blob storage feeds an accelerator training job, which writes checkpoints back to object/blob storage, with a dashed feedback path showing a restarted or preempted job resuming from the last checkpoint. Google Cloud, AWS, and Azure service names are labelled under each stage.](artefacts/training_workflow_diagram.png)

*Figure 1 — generated by `code/training_workflow_diagram.py` (the only executed code in this
chapter). Service names sourced from the table in Section 3.*

**Step 1 — initialise the SDK and point it at a staging bucket.** Same idea as setting an active
AWS profile/region before any `boto3` call: everything after this is scoped to a project, a region,
and a GCS bucket for job artifacts.

```python
from google.cloud import aiplatform

aiplatform.init(
    project="your-gcp-project-id",
    location="us-central1",
    staging_bucket="gs://your-gcp-project-id-training-staging",
)
```

**Step 2 — define the training job against a container and a training script.** `CustomTrainingJob`
wraps a training container/script; `container_uri` is a prebuilt framework image (or your own),
`script_path` is the entry point that runs inside it — the shape shown on the
[`google-cloud-aiplatform` PyPI page](https://pypi.org/project/google-cloud-aiplatform/) (checked
2026-09-02; the exact prebuilt image tag depends on which PyTorch/CUDA version you need — check
[Vertex AI's prebuilt-containers docs](https://docs.cloud.google.com/vertex-ai/docs/training/pre-built-containers)
for the current list rather than hardcoding one here). `gcsfs` is the library the training script
uses to read/write `gs://` paths as if they were local files; `2026.8.0` is its current PyPI release
([source: PyPI `gcsfs`](https://pypi.org/project/gcsfs/), checked 2026-09-02).

```python
job = aiplatform.CustomTrainingJob(
    display_name="resnet-finetune-job",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.<version>:latest",
    requirements=["gcsfs==2026.8.0"],
)
```

**Step 3 — run it against a GPU (or TPU), pointing the script at the GCS input and checkpoint
paths.** `machine_type` picks the VM shape; `accelerator_type`/`accelerator_count` pick the GPU
(current values per
[Vertex AI's compute-config docs](https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute),
checked 2026-09-02 — `NVIDIA_TESLA_T4` used here as a cost-conscious default; swap in
`NVIDIA_A100_80GB` or `NVIDIA_H100_80GB` for a larger job). `args` is how you hand the training
script its GCS input/output locations — the script itself reads/writes GCS paths with `gcsfs` or
the `google-cloud-storage` client, the same way it would read/write local paths.

```python
model = job.run(
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        "--train-data=gs://your-gcp-project-id-data/train-shards/",
        "--checkpoint-dir=gs://your-gcp-project-id-training-staging/checkpoints/",
        "--resume-from-latest",
    ],
    sync=True,
)
```

That `--resume-from-latest` flag is doing real work: it's the training script's own responsibility
to check the checkpoint prefix on startup and resume if a checkpoint exists, rather than assuming a
clean start. Section 6 explains why that's not optional.

**For a TPU instead of a GPU**, the worker pool spec swaps `accelerator_type`/`accelerator_count`
for a TPU `machine_type` and a topology — e.g. `machine_type="ct5lp-hightpu-4t"` with
`tpu_topology="4x4"` — per the
[TPU training docs](https://docs.cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm)
(checked 2026-09-02); everything else about the GCS input/checkpoint pattern is identical.

**The same shape in SageMaker** (`ModelTrainer`, the SDK v3 class that now replaces the older
per-framework `Estimator` classes — verified against
[SageMaker's checkpointing guide](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints-enable.html),
checked 2026-09-02): input data is an S3 URI passed to `.train()`, the accelerator is chosen via
`Compute(instance_type=...)` (e.g. `ml.p4d.24xlarge` for 8×A100,
[per the framework-estimator docs](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-framework-estimator.html),
checked 2026-09-02), and checkpoints go to S3 via `CheckpointConfig`:

```python
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, CheckpointConfig, InputData

model_trainer = ModelTrainer(
    training_image="<your-account-id>.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:<tag>-gpu",
    role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    compute=Compute(instance_type="ml.p4d.24xlarge", instance_count=1),
    base_job_name="resnet-finetune-job",
    checkpoint_config=CheckpointConfig(
        s3_uri="s3://your-bucket/resnet-finetune-job/checkpoints",
        local_path="/opt/ml/checkpoints",
    ),
)
model_trainer.train(
    input_data_config=[InputData(channel_name="training", data_source="s3://your-bucket/train-shards/")]
)
```

**The same shape in Azure ML** (`command()` + `MLClient`, SDK v2 — verified against
[Microsoft's own training how-to](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-model?view=azureml-api-2),
checked 2026-09-02): the compute target is a GPU `AmlCompute` cluster (an NC/ND-series size, e.g.
`Standard_NC24ads_A100_v4`), the input dataset is a Blob-backed `Input(type="uri_folder", path=...)`,
and the job's own code is responsible for writing checkpoints into the run's `outputs/` folder,
which Azure ML automatically syncs back to Blob Storage as the job runs:

```python
from azure.ai.ml import command, Input, MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

ml_client = MLClient(DefaultAzureCredential(), subscription_id="...", resource_group_name="...", workspace_name="...")

gpu_compute_target = "gpu-cluster"
try:
    ml_client.compute.get(gpu_compute_target)
except Exception:
    ml_client.compute.begin_create_or_update(
        AmlCompute(name=gpu_compute_target, size="Standard_NC24ads_A100_v4", min_instances=0, max_instances=2)
    ).result()

training_job = command(
    code="./src",
    command="python train.py --train-data ${{inputs.train_data}} --checkpoint-dir ./outputs/checkpoints",
    environment="<your-workspace-or-curated-pytorch-gpu-environment>@latest",
    inputs={"train_data": Input(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/train-shards/")},
    compute=gpu_compute_target,
)
ml_client.jobs.create_or_update(training_job)
```

All three follow the identical shape from Section 1: **object storage in → accelerator compute →
object storage out**, with the vendor filling in the class names and parameter shapes differently.

## 5. Data loading at scale — sharding, streaming, and the I/O bottleneck

An accelerator that costs several times a CPU VM's hourly rate is worthless if it spends half its
time idle waiting on data. This is the part of the workflow a Java engineer will recognise
immediately: it's a throughput problem, and the fix is the same one you'd reach for streaming a
large file into a service — **don't load it all into memory, and don't read it as one giant
sequential blob either.**

- **Sharding.** Instead of one enormous file, the dataset is split into many moderate-sized files
  (commonly tens to hundreds of MB each — think a `.tfrecord`, WebDataset `.tar`, or Parquet shard
  per few thousand examples) sitting in the bucket. Shards can be listed, shuffled, and read in
  parallel by multiple worker threads/processes, which is what actually keeps an accelerator fed —
  a single sequential stream from one huge file caps out at one connection's throughput.
- **Streaming instead of downloading first.** The training script opens a stream against the bucket
  (GCS, S3, or Blob Storage) and reads shards as it consumes them, rather than copying the whole
  dataset to local disk before starting. This is the cloud-storage equivalent of an
  `InputStream`/`Iterator` over a paginated API instead of `.collect(toList())`-ing everything up
  front — it bounds memory use and lets training start immediately instead of waiting on a full
  download.
- **The I/O bottleneck, concretely.** If your data pipeline (download → decode → decompress →
  augment → batch) can't keep pace with the accelerator's consumption rate, the accelerator sits
  idle between batches — you're paying full accelerator-hour price for a job that's actually
  bottlenecked on disk/network/CPU decode. The fix is almost always more parallel shard readers and
  more CPU worker processes for decode/augment (both `tf.data` and PyTorch's `DataLoader` support
  this directly), not a bigger or more expensive accelerator.

## 6. Pitfalls

- **Accelerator quotas.** GPUs and TPUs are not available by default the way a standard CPU VM is —
  every cloud gates them behind a per-project, per-region quota that starts at zero or a small
  number and has to be explicitly requested (and can take hours to days to approve). Requesting the
  quota is the first step, not something to discover after the job submission fails.
- **Egress cost.** Moving data *out* of a cloud region — downloading a trained model to your laptop,
  or worse, moving a dataset between two different clouds — is billed and is often the least
  visible line item in a training bill. Keep data, compute, and checkpoint storage in the same
  region, and only pull out the final artifact you actually need.
- **Tiny-batch underutilisation.** A GPU/TPU's throughput advantage comes from doing many examples'
  worth of matrix multiply in parallel; a batch size too small to fill that parallelism (common when
  code is ported unchanged from a CPU laptop setup) leaves most of the accelerator idle every step.
  If GPU/TPU utilisation metrics are low, check batch size and the data-loading pipeline (Section 5)
  before assuming you need a bigger accelerator.
- **Forgetting to checkpoint.** Spot/preemptible accelerator instances — the cheapest way to rent
  GPUs/TPUs — can be reclaimed by the cloud provider mid-job with little notice. A job that isn't
  periodically writing resumable checkpoints back to object storage (the right-hand loop in Figure
  1) loses all progress on reclaim and starts over from zero. This is the single most common way
  cloud training costs balloon: reruns of jobs that should have resumed instead of restarted.

## 7. When do you actually need this — honestly

For a cost-conscious engineer: **not as often as the marketing implies.** The worked examples
earlier in this curriculum — MNIST classification, a small text classifier fine-tune, the
transformer-internals walkthrough — all run acceptably on the CPU build from
[NOTE-ML-1](../../research/NOTE-ML-1-torch-install.md), on a laptop, for free. Reach for a cloud
accelerator when at least one of these is concretely true, not because a bigger tool feels more
serious:

- The model is large enough, or the dataset large enough, that a CPU run's wall-clock time is
  measured in days rather than minutes, and that time actually costs you something (a deadline, an
  iteration loop you need to run repeatedly).
- The dataset itself doesn't fit on local disk, so you need cloud object storage and streaming data
  loading (Section 5) regardless of what trains the model — at which point adding a rented GPU next
  to storage you already need is a much smaller incremental cost than provisioning it from nothing.
- You specifically need TPU-scale throughput for a matrix-multiply-dominated model training for
  weeks — a narrow, large-scale case (Section 2), not a default.

And when you do reach for it: start with the smallest GPU that fits the model (a single T4/L4 is
often enough for a fine-tune), use spot/preemptible capacity with checkpointing rather than
on-demand where your workload tolerates interruption, and treat egress and idle-accelerator time as
real line items to watch, not afterthoughts. The cloud accelerator earns its cost when it turns a
week of laptop time into an afternoon — not by being the default starting point.

## 8. Recap & what's next

- CPU training (this curriculum's default so far) breaks down on wall-clock time or dataset size,
  not because it's the "wrong" tool — a GPU/TPU only pays for itself once one of those two walls is
  actually hit (Section 1, Section 7).
- GPUs are general-purpose and framework-flexible; TPUs are a purpose-built matrix-multiply ASIC
  (128×128 systolic array MXU) that wins on large, long-running, matrix-dominated training and is
  Google-only (Section 2, grounded against
  [Cloud TPU introduction](https://docs.cloud.google.com/tpu/docs/intro-to-tpu)).
- The same three ingredients — object storage, a managed training service, accelerator hardware —
  exist under different names on every cloud (Section 3's table); the training-job *shape* (object
  storage in → accelerator compute → object storage out, Figure 1) is identical everywhere even
  though the SDK classes differ (Section 4).
- Efficient training at scale is a data-loading problem as much as a compute problem: shard the
  dataset, stream it instead of downloading it whole, and keep the pipeline fast enough to keep the
  accelerator fed (Section 5).
- Quotas, egress cost, tiny-batch underutilisation, and skipped checkpointing are the four ways this
  setup quietly wastes money — checkpointing in particular is what makes cheap preemptible
  accelerators usable at all (Section 6).

This closes the Cloud Environment Setup arc for Machine Learning. **Production Considerations**
picks up from here — serving the trained model, quantization, and the ongoing cost of keeping GPU/TPU
capacity around after training finishes, cross-referenced from the Data Science production material
where the concerns overlap.

---

### Environment note (for the architect)

Every service name, SDK entry point, GPU/TPU model, and instance-type string in Sections 2–4 was
verified live today (2026-09-02) against the cited official docs (Vertex AI compute-config,
prebuilt-containers, and TPU-training pages, the `google-cloud-aiplatform` and `gcsfs` PyPI pages,
Google's Cloud TPU introduction, AWS's EC2 accelerated-computing and SageMaker
checkpointing/estimator docs, and Microsoft's Azure ML training how-to plus the NC/ND VM-family
pages, and the `nca100v4-series` page confirming the Azure GPU size used in Section 4's snippet) —
all 14 citation URLs in this chapter return HTTP 200 as of this writing.
[NOTE-18](../../research/NOTE-18-managed-platforms.md) supplied the base managed-platform
service names and the pinned `google-cloud-aiplatform==2.1.0`; the GPU/TPU/accelerator specifics
were outside that NOTE's scope (as flagged in the spec) and were grounded fresh for this chapter
rather than assumed. The `NVIDIA_TESLA_K80` accelerator shown in some cached/older PyPI README
examples for `CustomTrainingJob` was deliberately **not** reused here — K80 is an old generation;
Section 4's snippet uses `NVIDIA_TESLA_T4`, a currently supported type per the Vertex AI
compute-config docs.
