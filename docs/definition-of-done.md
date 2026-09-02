# Definition of Done — chapter gate checklist

A chapter is DONE only when **every** box below is checked. No exceptions "to be fixed later".

## Fidelity to the spec
- [ ] Every learning objective in the approved `specs/SPEC-*.md` is delivered by the prose.
- [ ] Every worked example / dataset / artefact the spec lists is present.
- [ ] Anything cut from the spec is recorded in the spec's "Out of scope", not silently dropped.

## Grounded (the Haiku researcher's job)
- [ ] Every technical claim, metric definition, and API/library behaviour traces to a
      `research/NOTE-*.md` or an inline authoritative citation (link + date verified).
- [ ] Every package/library **version** named in the chapter or its `requirements` was verified
      against the live source (PyPI / official docs), not assumed from memory.
- [ ] Every **dataset link** resolves, and its licence / terms of use are stated.
- [ ] No claim rests on model memory alone.

## Runnable
- [ ] Every code snippet executes as written against the environment the chapter declares
      (imports complete, deps pinned), or is explicitly fenced as pseudocode.
- [ ] Every artefact (plot, table, metric output) reproduces from the committed code.
- [ ] Snippet compile check passes (`.claude/hooks/verify.sh` / `.claude/hooks/check_snippets.py`).

## Audience-fit
- [ ] Written for an experienced Java dev new to Python/ML: JVM/Java analogies where they clarify,
      no unexplained jargon, the "why" before the "how".
- [ ] The reader could reproduce the result on their own machine from the chapter alone.

## Links & hygiene
- [ ] Every external link resolves.
- [ ] No secrets committed; `.env.example` updated if config changed.

## Process
- [ ] One chapter per PR; PR body maps each acceptance criterion → its evidence (snippet / NOTE / artefact).
- [ ] Independent review by a **fresh** reviewer (not the writer) — sign-off recorded.
- [ ] Architect (Opus) merge approval.

## Escalate instead of forcing
Stop and ask the owner if a chapter's scope is ambiguous, a pedagogical cut needs a product decision,
a claim can't be grounded from available sources, or a planned dataset/tool is unavailable or paywalled.
