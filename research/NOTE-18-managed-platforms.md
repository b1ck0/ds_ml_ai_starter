# NOTE-18: Managed ML Platform Service Names and SDK Entry Points (2026)

**Answer:** 
Vertex AI (now Gemini Enterprise Agent Platform, as of 2026), Azure ML, and SageMaker maintain their core service names: notebooks/managed compute, pipelines, model registry, and endpoints. The google-cloud-aiplatform SDK is at v2.1.0 (released 2026-09-01); entry points remain aiplatform.init(), Pipeline, and Model.deploy().

**Evidence:**

| Cloud Platform | Managed Notebooks | Training Pipelines | Model Registry | Deployment Endpoints | SDK/Package | Version | Verified Date |
|---|---|---|---|---|---|---|---|
| **Vertex AI** (now Gemini Enterprise Agent Platform) | Workbench (notebooks.googleapis.com) | Vertex AI Pipelines | Model Registry | Endpoints | google-cloud-aiplatform | 2.1.0 | 2026-09-01 |
| **Azure ML** | Compute Instances | Pipelines / ML Jobs | Model Registry | Online Endpoints / Batch Endpoints | azureml-sdk | v2 (current) | 2026-03-15 |
| **AWS SageMaker** | SageMaker Studio | SageMaker Pipelines | Model Registry | SageMaker Endpoints | boto3/sagemaker-sdk | Integrated | 2026-08-15 |

**Source Documentation:**
- Vertex AI: https://docs.cloud.google.com/vertex-ai/docs/start/introduction-unified-platform (Overview, 2026)
- Vertex AI Python SDK: https://pypi.org/project/google-cloud-aiplatform/ (PyPI, verified 2026-09-01)
- Vertex AI Installation: https://docs.cloud.google.com/vertex-ai/docs/start/install-sdk (requires >=1.114.0)
- Google Generative AI SDK Docs: https://docs.cloud.google.com/python/docs/reference/vertexai/latest
- Azure ML Endpoints: https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints (Microsoft Learn, 2026)
- Azure ML Pipelines: https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk (Python SDK v2, 2026)
- SageMaker Services: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-deploy.html (AWS Documentation, 2026)

**Caveats / limits:**
- **Vertex AI Rebranding (2026):** Google rebranded Vertex AI to "Gemini Enterprise Agent Platform" as of 2026, but the underlying services (Workbench, Pipelines, Model Registry, Endpoints) continue with the same names and functionality.
- **Deprecated Vertex AI SDK modules (as of June 24, 2025):** The following modules are deprecated and will be removed on June 24, 2026: `vertexai.generative_models`, `vertexai.language_models`, `vertexai.vision_models`, `vertexai.tuning`, `vertexai.caching`. Use the Google Gen AI SDK instead.
- **SDK Entry Points:** The reference snippet should use `from google.cloud import aiplatform; aiplatform.init()` for initialization, and `aiplatform.gapic.aiplatform_v1beta1` or direct imports for pipeline and model APIs.
- **Python Version Requirement:** google-cloud-aiplatform 2.1.0 requires Python >=3.10.
- **Azure ML:** Uses SDK v2 API; earlier v1 API is deprecated (though some docs still reference both).

**Recommendation:**
- Pin `google-cloud-aiplatform==2.1.0` in requirements.txt (released 2026-09-01, current stable).
- Use the SDK entry points: `aiplatform.init(project=..., location=...)`, `PipelineJob()` for pipelines, and `Model.deploy()` for endpoints.
- Mark the code snippet as **reference (not executed in sandbox)** since cloud services cannot run locally.
- For Vertex AI, cite the official Google Cloud documentation dated 2026 (the overview and installation guides remain authoritative).
- Note the Gemini Enterprise Agent Platform rebranding in text if mentioning the service officially, but service names remain unchanged operationally.
- Azure ML and SageMaker service names have remained stable through 2026; no breaking changes documented.
