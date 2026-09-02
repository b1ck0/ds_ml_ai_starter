# Deploying agentic applications — GCP, AWS, Azure

*Agentic Engineering · Cloud Environment Setup · SPEC-AGENT-6*

**Nature of this chapter: grounded conceptual.** No cloud account exists in this sandbox, so every
`gcloud`/SDK block below is **reference only** — real, current service names and CLI/SDK shapes
pulled from each vendor's own documentation and verified live, not executed. Where you would
normally see a deployment's console output or a captured request/response, the text cites the
documented behaviour instead of inventing one. The one piece of code that actually runs is the
architecture diagram generator in `code/`. Every service name, version, and snippet shape below
traces to [research/NOTE-AGENT-5-cloud.md](../../research/NOTE-AGENT-5-cloud.md) or
[research/NOTE-18-managed-platforms.md](../../research/NOTE-18-managed-platforms.md) — nothing is
asserted from memory.

## 1. What & why — from localhost to production

Every agentic app you've built so far in this course — the MCP database server
([SPEC-AGENT-2](../../specs/SPEC-AGENT-2-mcp-database-query-layer.md)), the RAG pipeline over PDFs
([SPEC-AGENT-3](../../specs/SPEC-AGENT-3-rag-over-pdfs.md)), the invoice-extraction agent
([SPEC-AGENT-4](../../specs/SPEC-AGENT-4-invoice-agent.md)), the multi-agent Elders Tribunal
([SPEC-AGENT-5](../../specs/SPEC-AGENT-5-elders-tribunal.md)) — ran as a script or a local process:
a SQLite file on disk, a numpy array of embeddings in memory, an API key read from a `.env` file
only you could see. Production changes every one of those assumptions at once. The database has to
survive a process restart and be reachable from wherever the container runs. The vector store has to
hold more than ten documents and be reachable over the network, not imported as a local module. The
API key has to live somewhere a deployed container can fetch it without ever being checked into
source control or baked into an image layer. And the container itself has to be *somewhere* — a
managed runtime that can be reached over HTTPS, scaled, restarted, and observed.

For a Java engineer, this is the exact same shift as taking a Spring Boot app that runs `mvn
spring-boot:run` on a laptop and turning it into a service with a `Dockerfile`, a `DATABASE_URL`
environment variable instead of `localhost:5432`, and credentials pulled from a vault instead of
`application.properties`. Nothing about the *application logic* changes — the same MCP tool
boundary, the same RAG retrieval loop, the same LLM call. What changes is everything wrapped around
it: where it runs, where its data lives, where its secrets come from, and who's watching it once
real traffic hits it.

This chapter does four things, each grounded against current (2026) vendor documentation
(NOTE-AGENT-5): maps the agentic stack — container runtime, hosted LLM, vector store, secret
manager — to its managed-service name on each of GCP, AWS, and Azure (Section 2); walks one
concrete reference deployment end to end, on GCP, because "AlloyDB with pgvector for RAG SQL-native
vector search on GCP" and Cloud Run both have unusually good current documentation
(NOTE-AGENT-5's recommendation) (Section 3); reasons about the three things that quietly break a
production agent — cost, latency, observability (Section 4); and lays out the guardrails a
security-minded engineer adds before letting any of this touch real users or real data (Section 5).

## 2. Service mapping — the same four pieces, different names per cloud

Every agentic stack in this course needs exactly four managed pieces once it leaves your laptop: a
place to run the container, a hosted LLM to call, a place to store vectors for RAG, and a place to
keep secrets that isn't a `.env` file. All four exist, under different names, on every major cloud.
Verified live against each vendor's current documentation
([NOTE-AGENT-5](../../research/NOTE-AGENT-5-cloud.md), checked 2026-09-02):

| Layer | GCP | AWS | Azure |
|---|---|---|---|
| **Container / serverless runtime** | [Cloud Run](https://docs.cloud.google.com/run/docs) — serverless, near-zero cold start | Lambda (serverless, event-driven) + ECS/Fargate (long-running containers) | Container Apps — serverless Kubernetes |
| **Hosted LLM** | Vertex Generative API (Gemini models) — part of what Google rebranded the **Gemini Enterprise Agent Platform** in 2026; the SDK, package name, and API surface are unchanged (NOTE-AGENT-5 caveat 3) | Amazon Bedrock — hosts Claude, Mistral, LLaMA, and others behind one API | Azure OpenAI Service — GPT-4o, GPT-5, and others |
| **Managed vector store** | [Cloud SQL PostgreSQL](https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search) with the `pgvector` extension (0.8.1), or AlloyDB (pgvector + Google's ScaNN index) | Amazon RDS PostgreSQL (pgvector) or Aurora PostgreSQL (pgvector, tuned for write-heavy RAG ingestion) | Azure Database for PostgreSQL — pgvector, plus DiskANN and an `azure_ai` extension |
| **Secret manager** | [Secret Manager](https://docs.cloud.google.com/secret-manager/docs) — per-secret-version pricing, roughly $0.06/secret/month at scale | AWS Secrets Manager — roughly $0.40/secret/month plus API-call fees; built-in automatic rotation | Azure Key Vault — per-operation pricing, Standard or Premium (HSM-backed) tiers |

*(Full source list, per row: NOTE-AGENT-5 — GCP: [Cloud Run docs](https://docs.cloud.google.com/run/docs),
[AlloyDB vector search docs](https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search),
[Secret Manager docs](https://docs.cloud.google.com/secret-manager/docs), all checked 2026-09-02; AWS
and Azure service names verified against AWS's and Microsoft's own documentation, per NOTE-AGENT-5's
evidence trail.)*

Two read-throughs a Java engineer will recognise immediately:

- **The runtime row is the most familiar one.** Cloud Run, Fargate, and Container Apps are all
  "give us a container image, we run it and scale it" — the same shape as deploying a Spring Boot
  JAR to any managed container platform, just without provisioning the VM yourself. Lambda is the
  odd one out: it's built for short-lived, event-triggered invocations, not a long-running HTTP
  server, so it fits an agent that's woken by a queue message more than one serving continuous chat
  traffic.
- **The vector-store row is "just Postgres" everywhere.** If you already know `CREATE EXTENSION
  vector;` from the local pgvector setup in
  [SPEC-AGENT-0](../../specs/SPEC-AGENT-0-local-environment-setup.md) and the theory chapter's
  `<=>` cosine-distance operator ([theory.md §3](../Theory/theory.md)), you already know the
  production version too — it's the identical SQL, running against a managed instance instead of a
  local one. That portability is a real advantage over a bespoke vector database: the RAG retrieval
  code from [SPEC-AGENT-3](../../specs/SPEC-AGENT-3-rag-over-pdfs.md) doesn't need to change
  *shape* to move from a local pgvector file to Cloud SQL/RDS/Azure Postgres — only the connection
  string does.

The separate managed-ML-platform layer this course also touches —
[NOTE-18](../../research/NOTE-18-managed-platforms.md)'s Vertex AI / Azure ML / SageMaker mapping
for *training* jobs — is a different concern from the four rows above: NOTE-18 covers where you
*train* or fine-tune a model; this chapter covers where you *serve* one that's already hosted by the
vendor. NOTE-18 also confirms the same Vertex AI → Gemini Enterprise Agent Platform rebranding named
in the table above, and pins `google-cloud-aiplatform==2.1.0` (verified against
[PyPI](https://pypi.org/project/google-cloud-aiplatform/), checked 2026-09-01) if your deployment
also needs that SDK for anything beyond a plain HTTPS call to the hosted LLM endpoint.

## 3. A reference deployment — GCP: Cloud Run + Cloud SQL (pgvector) + Secret Manager

NOTE-AGENT-5 recommends GCP for the one worked reference deployment in this chapter, specifically
because Cloud Run's and AlloyDB's current docs are unusually clear to teach from — not because GCP
is objectively "the right cloud" (Section 2's mapping table exists precisely because all three are
viable, and the choice in practice comes down to your team's existing infrastructure and expertise,
per NOTE-AGENT-5's own recommendation). The three pieces wire together exactly the way Section 1
described: a container that can be reached over HTTPS, a Postgres instance holding vectors, and a
secret store the container reads from at startup — never a key baked into the image or checked into
git.

![Reference production architecture: a client sends an HTTPS request to a Cloud Run agent API (FastAPI) with rate limiting and input validation at the edge; the API calls Cloud SQL with pgvector for RAG retrieval and the Vertex Generative API for the hosted LLM; Secret Manager feeds it credentials at startup over a dashed line; the API exports spans to Cloud Trace over a dotted line; and any side-effecting tool call the model requests is routed through a human-in-the-loop approval queue, in red, before it reaches an external system.](artefacts/agentic_architecture_diagram.png)

*Figure 1 — generated by `code/agentic_architecture_diagram.py` (the only executed code in this
chapter). Service names and the reference-deployment shape are sourced from
[NOTE-AGENT-5](../../research/NOTE-AGENT-5-cloud.md)'s "Reference Deployment" and "Guardrails for
production" sections.*

Three components, in the order you'd actually stand them up:

**1 — The vector store.** Create a Cloud SQL PostgreSQL instance and enable pgvector inside it
(NOTE-AGENT-5, citing [AlloyDB's vector-search docs](https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search),
checked 2026-08-11 — pgvector 0.8.1 on both AlloyDB and Cloud SQL as of 2026-09-02):

```bash
gcloud sql instances create agent-vec-db \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --backup
```

```bash
gcloud sql connect agent-vec-db --user=postgres
# inside psql:
# CREATE EXTENSION vector;
# CREATE TABLE embeddings (id SERIAL, embedding vector(1536), metadata TEXT);
```

**2 — The secret.** Store the LLM API key in Secret Manager instead of an environment variable
baked into the deployment (NOTE-AGENT-5, citing the
[Secret Manager docs](https://docs.cloud.google.com/secret-manager/docs), checked 2026-09-02):

```bash
gcloud secrets create llm-api-key --data-file=- <<< "your-api-key-here"
```

Grant only the Cloud Run service account read access to that one secret — the least-privilege move
a Java engineer would recognise from scoping a database credential to exactly the schema a service
needs, not the whole instance:

```bash
gcloud secrets add-iam-policy-binding llm-api-key \
  --member=serviceAccount:PROJECT@appspot.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

Inside the running container, the API fetches that secret once at startup — not on every request,
which is exactly what the dashed "IAM-scoped secret read" arrow in Figure 1 shows (NOTE-AGENT-5's
Python shape for this call, `google-cloud-secretmanager`'s
`SecretManagerServiceClient.access_secret_version`, reference only — no live GCP project exists in
this sandbox to run it against):

```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(
    request={"name": "projects/PROJECT/secrets/llm-api-key/versions/latest"}
)
llm_api_key = response.payload.data.decode("UTF-8")
```

**3 — The container.** Deploy the agent API to Cloud Run, pointing it at the vector store and
wiring the secret in as a mounted environment variable rather than a plaintext one — `--set-env-vars`
for non-sensitive connection details, `--secret` for the key itself
(NOTE-AGENT-5, citing the [Cloud Run docs](https://docs.cloud.google.com/run/docs), checked
2026-09-02):

```bash
gcloud run deploy agent-api \
  --source=. \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --set-env-vars=VECTOR_DB_HOST=agent-vec-db,VECTOR_DB_PORT=5432,VECTOR_DB_USER=postgres \
  --secret=LLM_API_KEY=llm-api-key:latest
```

Cloud Run auto-sets a `PORT` environment variable and expects the container to listen on
`0.0.0.0:$PORT` — same contract whether the container is a FastAPI app wrapping this course's MCP
tool boundary ([SPEC-AGENT-2](../../specs/SPEC-AGENT-2-mcp-database-query-layer.md)) or the RAG
`answer.py` step from [SPEC-AGENT-3](../../specs/SPEC-AGENT-3-rag-over-pdfs.md) (NOTE-AGENT-5).
`--allow-unauthenticated` is shown here because it's what the reference tutorial uses; Section 5
comes back to why that flag is a decision to make deliberately, not a default to leave alone, once
real users (or real attackers) can reach the endpoint.

**A caveat worth carrying forward, not glossing over:** GCP's Secret Manager requires manual
rotation orchestration by default — AWS Secrets Manager and Azure Key Vault both support automatic
credential rotation for their managed database services out of the box (NOTE-AGENT-5, caveat 4).
If automatic rotation matters more to your team than GCP's cheaper per-secret pricing (NOTE-AGENT-5
puts Secret Manager at roughly $0.06/secret/month at scale, against AWS's roughly $0.40 plus
call fees), that's a real reason the mapping table in Section 2 isn't just cosmetic — it's the axis
you'd actually weigh a cloud choice on.

For real production deployment beyond this reference sketch — networking, IAM policy design, CI/CD
wiring — NOTE-AGENT-5 is explicit that the right next step is each vendor's own Terraform provider
docs, not a bespoke hand-rolled script; this chapter's snippets are the CLI-level shape to recognise,
not a template to copy into a pipeline unmodified.

## 4. Cost, latency, and observability

**Cost has two independent meters, and they don't move together.** Infrastructure cost (Cloud Run
vCPU-seconds, Cloud SQL instance-hours, Secret Manager's per-secret and per-access fees) behaves the
way every cloud bill you've seen before behaves — proportional to provisioned capacity and request
volume. **Token cost is a separate meter entirely**, billed per input and output token by the LLM
provider, and it scales with something infrastructure billing doesn't see at all: how much text you
put in the prompt. [theory.md §2](../Theory/theory.md) already named this for a single LLM call —
every retrieved RAG chunk, every turn of conversation history, every system-prompt instruction is a
line item. A production deployment multiplies that per-call cost by request volume, and the Elders
Tribunal pattern from [SPEC-AGENT-5](../../specs/SPEC-AGENT-5-elders-tribunal.md) multiplies it
again: several elder agents, each holding their own LLM call across multiple debate rounds, means
the token bill for one user-facing "answer" is the sum of every one of those calls, not one call's
worth. Watch the token meter separately from the infrastructure meter — a Cloud Run bill that looks
perfectly reasonable can sit next to an LLM API bill that's ten times larger, because they're
metering entirely different things.

**Latency has a similar split: infra cold-start vs. model time-to-first-token.** Cloud Run's own
cold-start behaviour is near-zero for most workloads — NOTE-AGENT-5 puts first-invocation startup
at roughly 100ms plus network latency, which is the container-runtime part of the latency budget.
The larger, less controllable part is the hosted LLM's own response time. NOTE-AGENT-5's 2026
performance comparison across the three hosted-LLM services (time-to-first-token, a proxy for how
quickly a user sees the first token of a streamed response):

| Provider | Time-to-first-token |
|---|---|
| Azure OpenAI Service | ~180ms |
| Google Cloud Vertex (Gemini) | ~210ms |
| AWS Bedrock | ~245ms |

(NOTE-AGENT-5, "Performance Notes (2026)"; NOTE-AGENT-5 additionally notes Gemini models are
fastest on *total* completion time due to efficient token generation once the first token arrives —
time-to-first-token and total-completion-time are two different numbers worth tracking separately in
your own metrics.) For the vector-search leg of a RAG-backed agent, NOTE-AGENT-5 notes AlloyDB's
ScaNN index outperforms standard pgvector, and Aurora PostgreSQL is specifically tuned for
write-heavy RAG ingestion workloads — relevant if your bottleneck turns out to be retrieval latency
rather than the LLM call itself.

**Observability is what turns "the agent did something wrong" into "here's exactly which call and
which retrieved chunk caused it."** Every major cloud ships a built-in tracing service for this: GCP
Cloud Trace, AWS X-Ray, Azure Monitor (NOTE-AGENT-5, "Guardrails for production," point 4) — the
dotted arrow in Figure 1. The reason this matters specifically for an agent, more than for a typical
REST service: a single user request can fan out into several LLM calls, a vector-store query, and a
tool call, and unlike a stack trace from an exception, an LLM's output gives you no automatic signal
about *which* of those steps produced a bad answer. Tracing every span — the retrieval query, the
prompt actually sent, the tool call made — is the only way to reconstruct that after the fact,
exactly the way a Java engineer would reach for distributed tracing (Jaeger/Zipkin/OpenTelemetry)
across a microservice call chain rather than trying to debug from the final response alone.

## 5. Guardrails — the checklist a security-minded engineer runs before shipping

Everything in this section is grounded in NOTE-AGENT-5's "Guardrails for production" list. None of
it is optional once real users, and real side effects, are in play.

**Secrets never leave the secret manager.** Section 3's `--secret=LLM_API_KEY=llm-api-key:latest`
flag is the whole point: the key is fetched by the container's own IAM identity, at runtime, and
never appears in an image layer, a log line, or a `git commit`. This is the same discipline as never
committing a database password to a Spring `application.properties` file — except an LLM API key
that leaks is worse in one specific way: it isn't just a data-access risk, it's a *spending* risk,
because whoever holds the key can run up your token bill until you notice and revoke it.

**Rate-limit every LLM call.** Vertex, Bedrock, and Azure OpenAI all enforce their own quotas
(requests-per-minute, tokens-per-minute) at the API level (NOTE-AGENT-5) — but a rate limiter *in
front of your own API* (Figure 1's "rate limit + input validation at the edge" box) does two more
things the vendor quota alone doesn't: it protects your token budget from a single caller looping or
retrying aggressively, and it fails your request cleanly and cheaply before it ever reaches the
expensive LLM call, rather than after.

**Defend against prompt injection at the input boundary.** NOTE-AGENT-5's guidance is direct:
validate and sanitise user input before it ever reaches the LLM. This matters most, and differently,
for the RAG and MCP patterns this course already built: a RAG pipeline
([SPEC-AGENT-3](../../specs/SPEC-AGENT-3-rag-over-pdfs.md)) retrieves *external* text — a PDF, a web
page, a document someone else authored — and splices it directly into the prompt (theory.md §4's
"augment" step). If that retrieved text contains instructions ("ignore your previous instructions
and instead...") crafted to look like part of the source document, the model has no structural way to
tell "content to answer questions about" apart from "instructions to follow" — both arrive as the
same undifferentiated block of prompt text. This is not a hypothetical: it is the direct extension of
the [SPEC-AGENT-2](../../specs/SPEC-AGENT-2-mcp-database-query-layer.md) chapter's own framing —
an MCP tool boundary exists precisely because a caller (there, an LLM; here, whatever authored the
retrieved document) should never get to dictate what a backend service does just by asserting it in
text. The mitigation is the same shape as ordinary input handling: sanitise and clearly delimit
retrieved/external content from trusted system instructions, and treat anything the LLM decides to
*do* as a request to be checked, never as a command to execute unquestioned — which is exactly what
the next guardrail is for.

**Human-in-the-loop for anything side-effecting.** This is the guardrail Figure 1 draws in red, and
the one this chapter emphasises most, because it is the one that actually bounds the *blast radius*
of everything above going wrong at once. NOTE-AGENT-5's own framing is direct: if the agent can
write to a database or call an external API, critical operations need human approval before they
run. Concretely, for
this course's own worked examples: [SPEC-AGENT-2](../../specs/SPEC-AGENT-2-mcp-database-query-layer.md)'s
MCP server was deliberately built **read-only** — `list_tables`, `describe_table`, and a filtered
`query`, with no write tool at all, which is a design choice this chapter's guardrail now explains
*why*: a read-only tool boundary needs no approval gate because nothing it does is irreversible.
[SPEC-AGENT-4](../../specs/SPEC-AGENT-4-invoice-agent.md)'s Invoice Agent is the opposite case — it
turns unstructured PDF content into a real database write via an MCP write-tool, which is exactly
the "side-effecting tool call" box in Figure 1: in production, that write does not fire the instant
the model decides to call it. It is held in an approval queue until a person confirms it, the same
way a junior engineer's production database migration gets a second pair of eyes before it runs, not
after. The reasoning holds even when the model's confidence looks high: an LLM's fluency is not
correlated with correctness (theory.md §1), and a side effect — a database row written, an email
sent, a payment initiated — cannot be silently undone the way a bad *read* can simply be re-queried.

## 6. Pitfalls

**Leaking keys.** The most common way this happens is not a hacked secret manager — it's a
`--allow-unauthenticated` Cloud Run flag left in from a demo, an API key pasted into a log statement
for debugging and never removed, or a key committed to a public repo's `.env` file during a rushed
fix. Section 5's rule — secrets fetched by IAM identity, never baked in or logged — is defence
against the boring, common case, not just the sophisticated one.

**Unbounded token cost.** Section 4 already separated the two cost meters; the failure mode is
deploying without a per-request or per-user token budget and discovering the bill only at the end of
the month. A single retry loop with no cap, a multi-agent debate pattern like
[SPEC-AGENT-5](../../specs/SPEC-AGENT-5-elders-tribunal.md) run with no round limit, or a RAG
pipeline that stuffs the full top-k retrieved set into every prompt regardless of relevance
(theory.md §2's "over-stuffing" pitfall) are all silent cost multipliers that produce no error, no
alert, and no signal until someone reads the invoice.

**No tracing.** Skipping Section 4's observability step doesn't fail loudly either — the agent keeps
answering requests, just as a service without logging keeps serving requests. What's missing only
becomes visible the first time an answer is wrong and there's no way to reconstruct which retrieved
chunk, which tool call, or which of several LLM calls in a multi-agent chain actually produced it.
Wire up tracing (Cloud Trace/X-Ray/Azure Monitor, per Section 4) before the first real user, not
after the first real incident.

**Trusting agent side effects.** The single most important guardrail in this chapter, restated as
its own failure mode: any production deployment that lets an LLM's tool call execute a
database write, an external API call, or any other irreversible action *without* the human-in-the-loop
gate from Section 5 is one confidently-wrong model response away from an incident that can't be
undone by re-asking the question. The read-only MCP boundary from
[SPEC-AGENT-2](../../specs/SPEC-AGENT-2-mcp-database-query-layer.md) and the approval-gated write
tool this section built on top of the Invoice Agent's write path
([SPEC-AGENT-4](../../specs/SPEC-AGENT-4-invoice-agent.md)) are the two concrete patterns this
course has already shown you; production deployment is the point where skipping the gate stops being
a shortcut and starts being the actual risk.

## Recap & what's next

Moving an agentic app to production changes *where* four things live — the container runtime, the
hosted LLM, the vector store, and secrets — never the application logic built in
[SPEC-AGENT-2](../../specs/SPEC-AGENT-2-mcp-database-query-layer.md) through
[SPEC-AGENT-5](../../specs/SPEC-AGENT-5-elders-tribunal.md) (Section 1). All three major clouds offer
the same four managed pieces under different names, and the vector-store row is close to portable
by default because it's "just Postgres" everywhere (Section 2). This chapter's one concrete reference
deployment — Cloud Run + Cloud SQL with pgvector + Secret Manager (Figure 1, Section 3) — is a
pattern, not a template to copy verbatim; real production IaC belongs in each vendor's own Terraform
docs. Cost splits into an infrastructure meter and a separate, request-volume-scaling token meter;
latency splits into infra cold-start and hosted-LLM time-to-first-token; and neither is visible after
the fact without tracing wired up from day one (Section 4). And the guardrails — secrets never
leaving the secret manager, rate limits in front of the LLM call, input sanitisation against prompt
injection, and above all a human-in-the-loop gate on every side-effecting tool call — are what keep
a production agent's mistakes small and reversible instead of silent and permanent (Section 5).

This closes the Agentic Engineering track's Cloud Environment Setup section. **Production
Considerations** picks up from here with a deeper look at the same cost/latency concerns under
sustained real traffic, systematic evaluation of agent outputs (not just spot-checking), and data
privacy for RAG corpora that may contain content you don't control.

---

### Environment note (for the architect)

Every service name, version, cost figure, and latency number in this chapter traces to
[NOTE-AGENT-5-cloud.md](../../research/NOTE-AGENT-5-cloud.md) (checked 2026-09-02) or
[NOTE-18-managed-platforms.md](../../research/NOTE-18-managed-platforms.md) (checked 2026-09-01/2026-09-02)
— no claim rests on model memory. All four inline external citation URLs (Cloud Run docs, AlloyDB
vector-search docs, Secret Manager docs, and the `google-cloud-aiplatform` PyPI page) were checked
live today (2026-09-03) and return HTTP 200. The Python `secretmanager` snippet in Section 3 and the `gcloud`/SQL CLI
snippets throughout are reference-only, reproduced from NOTE-AGENT-5's own evidence trail — none are
executed in this sandbox, since no GCP project or credentials exist here; the snippet-compile gate
(`check_snippets.py`) still byte-compiles the Python block to catch a syntax error, which is the
extent of what "runnable" means for reference code in a GROUNDED-CONCEPTUAL chapter per its spec.
The only code actually executed and reproduced for this chapter is
`code/agentic_architecture_diagram.py`, run with `.venv/Scripts/python.exe`.
