# NOTE-AGENT-5: Cloud Services for Agentic Applications — 2026 Service Mapping and Reference Deployment

**Answer:**
Current (2026) managed services verified across GCP/AWS/Azure: container runtime (Cloud Run, ECS/Lambda/Fargate, Container Apps), hosted LLM (Vertex Generative API, Amazon Bedrock, Azure OpenAI Service), managed vector DB (Cloud SQL/AlloyDB pgvector, AWS Aurora/RDS pgvector, Azure Postgres pgvector), secret manager (Secret Manager, Secrets Manager, Key Vault). Reference deployment: GCP container (Cloud Run) + vector store (Cloud SQL with pgvector) + secret manager (Secret Manager).

**Evidence:**

*Service Mapping Table (verified 2026-09-02):*

| Layer | GCP | AWS | Azure |
|-------|-----|-----|-------|
| **Container Runtime** | Cloud Run (serverless, ~0ms cold start) | Lambda (serverless events) + ECS/Fargate (long-running containers) | Container Apps (serverless Kubernetes) |
| **Hosted LLM** | Vertex Generative API (Gemini models) | Amazon Bedrock (Claude, Mistral, LLaMA, etc.) | Azure OpenAI Service (GPT-4o, GPT-5, etc.) |
| **Managed Vector Store** | Cloud SQL PostgreSQL (pgvector 0.8.1) + AlloyDB (pgvector + ScaNN) | AWS RDS PostgreSQL (pgvector) + Aurora PostgreSQL (pgvector, optimized for writes) | Azure Database for PostgreSQL (pgvector + DiskANN + azure_ai extension) |
| **Secret Manager** | Google Secret Manager (per-version pricing, ~$0.06/secret/month at scale) | AWS Secrets Manager (~$0.40/secret/month + API call fees) | Azure Key Vault (per-operation pricing, Standard or Premium HSM-backed) |

Source Documentation:
- GCP: https://docs.cloud.google.com/run/docs, https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search, https://docs.cloud.google.com/secret-manager/docs (verified 2026-09-02)
- AWS: AWS documentation (https://docs.aws.amazon.com, verified via comparisons 2026-09-02)
- Azure: https://azure.microsoft.com/en-us/products/postgresql, https://learn.microsoft.com (verified 2026-09-02)
- Comparative reviews: https://techsy.io/en/blog/aws-vs-azure-vs-google-cloud, https://tech-champion.com/cloud-computing/postgresql-in-the-cloud-the-definitive-2026-comparison-of-rds-aurora-cloud-sql-and-alloydb/ (2026)

*Performance Notes (2026):*
- **Time-to-first-token**: Azure OpenAI Service (180ms) > Google Cloud Vertex (210ms) > AWS Bedrock (245ms)
- **Total completion**: Google Cloud Gemini models fastest due to efficient token generation
- **Vector search**: AlloyDB with ScaNN outperforms standard pgvector; Aurora PostgreSQL optimized for write-heavy RAG workloads

*Reference Deployment (GCP — Container API + Managed Vector Store + Secrets):*

**Architecture:**
1. **API Container** (Cloud Run, FastAPI or similar)
   - Endpoint: `gcloud run deploy my-agent-api --image gcr.io/PROJECT/agent-api:latest --region us-central1 --allow-unauthenticated --set-env-vars=VECTOR_DB_HOST=...`
   - PORT env var auto-set; listens 0.0.0.0:8080
   - Source: https://docs.cloud.google.com/run/docs

2. **Vector Database** (Cloud SQL or AlloyDB + pgvector)
   - Create instance: `gcloud sql instances create agent-db --database-version=POSTGRES_16 --tier=db-f1-micro --region=us-central1`
   - Enable pgvector: Connect and run `CREATE EXTENSION vector;`
   - Version: pgvector 0.8.1 (AlloyDB; Cloud SQL 0.8.1 as of 2026-09-02)
   - Source: https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search (verified 2026-08-11)

3. **Secrets Management** (Google Secret Manager)
   - Store LLM API key: `gcloud secrets create llm-api-key --data-file=-`
   - Retrieve in Cloud Run: `from google.cloud import secretmanager; client = secretmanager.SecretManagerServiceClient(); secret = client.access_secret_version(request={'name': 'projects/PROJECT/secrets/llm-api-key/versions/latest'})`
   - Grant Cloud Run service account access: `gcloud secrets add-iam-policy-binding llm-api-key --member=serviceAccount:PROJECT@appspot.gserviceaccount.com --role=roles/secretmanager.secretAccessor`
   - Source: https://docs.cloud.google.com/secret-manager/docs (verified 2026-09-02)

**Minimal Terraform-like CLI snippet (reference only, not executed):**
```bash
# Create VPC (optional but recommended for security)
gcloud compute networks create agent-network --subnet-mode=custom
gcloud compute networks subnets create agent-subnet --network=agent-network --range=10.0.0.0/24 --region=us-central1

# Create Cloud SQL instance with pgvector
gcloud sql instances create agent-vec-db \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --network=agent-network \
  --backup

# Enable pgvector extension
gcloud sql connect agent-vec-db --user=postgres
# In psql: CREATE EXTENSION vector; CREATE TABLE embeddings (id SERIAL, embedding vector(1536), metadata TEXT);

# Create secret for LLM API key
gcloud secrets create llm-api-key --data-file=- <<< "your-api-key-here"

# Deploy Cloud Run API
gcloud run deploy agent-api \
  --source=. \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --set-env-vars=VECTOR_DB_HOST=agent-vec-db,VECTOR_DB_PORT=5432,VECTOR_DB_USER=postgres \
  --secret=LLM_API_KEY=llm-api-key:latest
```

Source: https://dev.to/suhas_mallesh/alloydb-ai-with-pgvector-for-rag-sql-native-vector-search-on-gcp-with-terraform-2hbe (tutorial, 2026)

**Caveats / limits:**

1. **Cloud Run cold-start**: Near-zero for most workloads; first invocation ~100ms startup + network latency.

2. **pgvector extension versions**: AlloyDB ships pgvector 0.8.1; Cloud SQL 0.8.1 (as of 2026-09-02). AWS RDS typically lags by 1–2 minor versions; verify version before relying on newest features (e.g., HNSW indexing).

3. **Managed LLM availability**: Bedrock now supports agent capabilities (early 2026); Azure OpenAI Service integrates with Azure AI Foundry for observability; Vertex Generative API is part of the Gemini Enterprise Agent Platform (rebranded 2026 but service names unchanged).

4. **Secrets rotation**: Azure Key Vault and AWS Secrets Manager support automatic rotation for RDS/Cosmos credentials. GCP Secret Manager requires manual rotation orchestration (or GKE secrets auto-rotation add-on, now GA 2026).

5. **Cost**: GCP Secret Manager cheapest at scale (~$0.06/secret/month); AWS Secrets Manager most expensive for many secrets but includes rotation automation. Pricing varies significantly by call volume vs. secret count.

6. **Regional replication**: GCP Secret Manager auto-replicates within region by default; AWS and Azure require explicit multi-region setup for DR.

**Recommendation:**

1. **Choose cloud based on existing infrastructure and team expertise.** All three clouds are viable for agentic apps (2026).

2. **For GCP reference code:**
   - Use official CLI docs: https://docs.cloud.google.com/run/docs, https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search
   - Cite examples dated August 2026 or later (released 2026-08-11 for AlloyDB, 2026-09-02 for release notes)
   - Document that code is reference (no local execution without GCP project credentials)

3. **For teaching:**
   - Show the service mapping table (cross-cloud alignment)
   - Provide ONE cloud's reference deployment (GCP recommended for readability; Cloud Run has excellent docs)
   - Link to official IaC (Terraform docs on cloud provider sites, not bespoke snippets) for production deployment
   - Emphasize secrets management as the critical security gate (cover in guardrails section)

4. **Guardrails for production (cross-cloud):**
   - Never commit API keys; always use secret manager
   - Rate-limit LLM calls (Bedrock quotas, Vertex API quotas, Azure OpenAI TPM limits)
   - Prompt injection defence: validate/sanitize user input before passing to LLM
   - Human-in-the-loop for side-effecting tools (e.g., if agent can write to database or call external APIs, require human approval for critical operations)
   - Tracing/observability: GCP Cloud Trace, AWS X-Ray, Azure Monitor (built-in to managed services)

**Date checked:** 2026-09-02
