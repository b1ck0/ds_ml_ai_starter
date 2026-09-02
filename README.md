# ds_ml_ai_starter

A hands-on **textbook / crash course** in Data Science, Machine Learning, Agentic Engineering, and
AI-assisted SDLC — written for an experienced backend engineer (15+ years Java) who is new to Python
and ML. Every concept lands as runnable code and a visible artefact, not just prose.

## The four parts
1. [Data Science](Data%20Science/) — stats, EDA, regression, classification, forecasting, MLOps.
2. [Machine Learning](Machine%20Learning/) — neural nets, CV, NLP, LLMs with PyTorch/TensorFlow.
3. [Agentic Engineering](Agentic%20Engineering/) — vector DBs, RAG, MCP, multi-agent apps.
4. [AI-assisted SDLC](AI-assisted-sdlc/) — driving agents with specs, gates, hooks, and subagents.

## How this repo is built (it eats its own dog food)

This is a **spec-driven, grounded, multi-model authoring** project — itself a live example of the
AI-assisted SDLC part.

| Role | Model | Does |
|---|---|---|
| Architect | **Opus 4.8** (main session) | Scopes each chapter into a spec, reviews, gates, merges |
| Writer | **Sonnet 4.6** | Writes one approved chapter: prose + runnable code + artefacts |
| Researcher | **Haiku** | Searches the web to ground every claim, version, and dataset link |
| Reviewer | fresh **Sonnet** | Independent QA before merge |

The rules live in [`CLAUDE.md`](CLAUDE.md); the workflow in
[`docs/architecture.md`](docs/architecture.md); the backlog in
[`docs/curriculum.md`](docs/curriculum.md); the audience/voice in
[`docs/style-guide.md`](docs/style-guide.md); the exit gate in
[`docs/definition-of-done.md`](docs/definition-of-done.md).

**Golden rule:** no chapter is written without an approved `specs/SPEC-*.md`, and nothing ships
ungrounded — every fact traces to a `research/NOTE-*.md` or an inline citation.

## Driving it
1. Pick a backlog item from `docs/curriculum.md` → the architect scopes it into `specs/SPEC-*.md`
   (`.claude/skills/chapter-scoper`) and gets sign-off.
2. The architect briefs the **Haiku researcher** to ground every external claim → `research/NOTE-*.md`.
3. The architect dispatches the **Sonnet writer** against the one approved spec.
4. A **fresh Sonnet** reviews against `docs/definition-of-done.md`; the architect merges.

First scoped chapter (draft): [`SPEC-DS-1` — Hypothesis Testing & EDA](specs/SPEC-DS-1-hypothesis-testing-and-eda.md).

## Local setup
Python 3.11+. Create a virtualenv and install the shared tooling; heavy per-subject deps (PyTorch,
etc.) are installed per chapter as its spec declares.
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
