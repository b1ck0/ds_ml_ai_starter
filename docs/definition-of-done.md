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

## Renders on GitHub
GitHub runs Markdown over `$…$`/`$$…$$` **before** MathJax, so it silently corrupts LaTeX. The
always-safe venue for a displayed equation is a fenced `math` block (a code fence tagged `math`),
which is not Markdown-processed.
- [ ] `.claude/hooks/check_markdown_render.py` passes — it is also run automatically on every push by
      the `pre-push` git hook, which blocks a push that fails it.
- [ ] **Every displayed equation is a fenced `math` block, not `$$…$$`.** (In `$$`/`$`, Markdown eats
      the backslash in `\_ \# \{ \} \~` → `\text{one\_hot}` becomes `one_hot` → "'_' allowed only in
      math mode", and multi-line content is mangled.)
- [ ] Inline `$…$` is simple and on ONE line: no `\\`/matrices (they render raw, worst in a table
      cell — use a fenced `math` block), no `<` before a letter (write `\lt`), no `\_ \# \{ \}`
      escapes, no split across a line break; `\text{}` has no unescaped `_ ^ # % & ~`; braces balance.
- [ ] No `\( \) \[ \]` delimiters, no `align`/`equation` environments (use a `math` fence + `aligned`),
      no user macros (`\newcommand`/`\def`) — GitHub supports none of them.
- [ ] Every `math`/`mermaid` fence and every `$…$` was eyeballed as rendered (the checker is necessary,
      not sufficient); Mermaid node labels with `(`/`)` are `"double-quoted"`.

## Audience-fit
- [ ] Written for an experienced Java dev new to Python/ML: JVM/Java analogies where they clarify,
      no unexplained jargon, the "why" before the "how".
- [ ] The reader could reproduce the result on their own machine from the chapter alone.

## Links & hygiene
- [ ] Every external link resolves.
- [ ] No secrets committed; `.env.example` updated if config changed.

## Repository coherence (standing rule)
Adding, renaming, or rescoping a chapter is not done until **every related document** is updated in
the same change, so the repo stays internally consistent:
- [ ] `docs/curriculum.md` (backlog / chapter list) reflects the change.
- [ ] The relevant `<subject>/README.md` section listing includes the chapter in the right order.
- [ ] Sibling cross-links updated — prerequisites, "what's next" pointers, and any "you are here" maps.
- [ ] `docs/architecture.md` updated if the repository shape changed.
- [ ] `NN-` numbering stays contiguous and matches the intended reading order.

## Process
- [ ] One chapter per PR; PR body maps each acceptance criterion → its evidence (snippet / NOTE / artefact).
- [ ] Independent review by a **fresh** reviewer (not the writer) — sign-off recorded.
- [ ] Architect (Opus) merge approval.

## Escalate instead of forcing
Stop and ask the owner if a chapter's scope is ambiguous, a pedagogical cut needs a product decision,
a claim can't be grounded from available sources, or a planned dataset/tool is unavailable or paywalled.
