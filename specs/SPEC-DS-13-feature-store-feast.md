# SPEC-DS-13: Feature Store — slow vs fast features with Feast

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-5/DS-6, SPEC-DS-12

## Intent
Explain the problem a feature store solves: training uses batch ("slow") features while online serving
needs the same features fresh and low-latency ("fast"), and the two must agree to avoid train/serve
skew. Teach Feast's offline/online split and its unified SDK on a small local example.

## Learning objectives
- LO1 — Define train/serve skew and why one feature definition must serve both offline and online.
- LO2 — Explain slow (batch) vs fast (online/real-time) features with examples.
- LO3 — Define feature views in Feast, materialise to an online store, and retrieve historical (training) and online (serving) features via one SDK.
- LO4 — Reason about where a feature store fits in a production ML system (vs computing features ad-hoc).

## Scope
In: Feast concepts (entities, feature views, offline vs online store), a local file/sqlite demo, get_historical_features vs get_online_features.
Out: cloud-backed stores (Redis/BigQuery/DynamoDB) beyond a mention, streaming feature pipelines (mention + link).

## Outline
1. What & why — the skew problem; the same feature computed two ways drifts.
2. Slow vs fast features — batch aggregates vs real-time lookups.
3. Feast setup — repo, entity, feature view, offline store (parquet), online store (sqlite).
4. Retrieve training features (point-in-time correct) and online features via the SDK.
5. Where it fits — a diagram of offline training + online serving sharing definitions.
6. Pitfalls — point-in-time leakage, materialisation staleness, over-adopting a store too early.

## Claims to ground (Haiku, before writing) — IMPORTANT
- [ ] Verify the current Feast version on PyPI and that a LOCAL (file offline + sqlite online) demo installs and runs in a sandbox on the target Python.
- [ ] Verify the current Feast API/workflow: feature_store.yaml, Entity, FeatureView, feast apply, materialize, get_historical_features, get_online_features. Confirm the current concepts (Feast has changed APIs across versions).
- [ ] Confirm the point-in-time-correct join concept against Feast docs.

## Assets to produce
- Prose: "Data Science/Worked Examples/feature-store-feast.md"
- Code: "Data Science/Worked Examples/code/feast_demo/" (repo files + a driver script)
- Artefacts: an offline vs online architecture diagram (matplotlib/SVG); a captured retrieval output.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the local Feast demo actually applies + materialises + retrieves; snippet-check passes. If Feast can't run in-sandbox, ESCALATE before writing (don't fake it). AC3 — Feast version + API grounded. AC4 — train/serve skew made concrete; the "unified SDK" value clear.

## Gates
Entry: approved; grounding (esp. installability) landed. Exit: DoD checklist.
