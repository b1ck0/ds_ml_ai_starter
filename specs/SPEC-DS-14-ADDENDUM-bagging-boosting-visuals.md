# SPEC-DS-14-ADDENDUM: Bagging & boosting — step-by-step mechanism visuals

**Status:** approved
**Subject:** Data Science
**Section:** Theory (amends `01-data-science/01-theory/01-theory-overview.md` §4, SPEC-DS-14)
**Routing:** writer=Sonnet 4.6 · research=Haiku (none/light) · review=Sonnet (fresh) · architect=Opus 4.8

## Why this addendum
DS-14 §4 already *names* bagging vs boosting in a comparison table and a taxonomy node
(`ENS → BAG / BOOST`, ~lines 291–343). What the owner asked for and the chapter lacks is a
**step-by-step visual of how each one actually works** — the same "watch the mechanism unfold"
treatment the cross-validation diagram gives splitting. Boosting in particular is *used* elsewhere
(`HistGradientBoosting` in DS-10/DS-11) but its mechanism is never drawn. Add the mechanism, right
after the existing bagging-vs-boosting table in §4, before §5.

## What to add (in §4, after the comparison table)
1. **Bagging, step by step.** A Mermaid diagram + a tiny numeric walk-through: from one training set,
   draw N bootstrap samples (with replacement — show which rows repeat/drop for one small sample),
   fit one model per sample **in parallel**, then **average / majority-vote**. Show, with a few real
   numbers, how independent errors partly cancel → variance drops (tie to the bias–variance section
   already in §5.2, and to the concrete DS-8 result where `BalancedBaggingClassifier` fixed a noisy
   undersample). Style it like the existing cross-validation / K-fold diagrams.
2. **Boosting, step by step.** A Mermaid diagram + a tiny numeric walk-through of the *sequential*
   loop: fit a weak learner (a stump) → find what it got wrong (misclassified points / residuals) →
   **reweight** (AdaBoost) or **fit the residual** (gradient boosting) → fit the next stump on the
   corrected target → add it in with a weight → repeat, watching the training error shrink each round
   (iterate-visibly, "closer to perfection each iteration"). Make the sequential dependency visually
   obvious (each stage feeds the next), in contrast to bagging's parallel fan-out.
3. **One-glance contrast.** A short side-by-side: parallel + vote (variance↓) vs sequential + correct
   (bias↓), reinforcing the table that's already there — now with the mechanism the table summarises.

Keep it conceptual and hand-checkable. A tiny runnable snippet (a handful of points, 2–3 stumps with
`sklearn`) that prints the shrinking error is welcome but optional; a clear worked-by-hand numeric
example is sufficient. If a snippet is added it must pass the snippet gate.

## Constraints
- **Extend, don't rewrite.** Preserve existing prose and code blocks byte-for-byte; add the two
  mechanism diagrams + walk-throughs and the contrast. House style throughout (real numbers,
  problem-first, plain language before notation, Java analogy where it helps).
- Visuals are GitHub-native Mermaid + LaTeX. Escape specials inside `\text{…}` (e.g. residual/weight
  update formulas); wrap any Mermaid node label containing `(` or `)` in `"double quotes"`.
- Style references to match: the existing bagging Mermaid in
  `01-data-science/03-worked-examples/08-class-imbalance.md` (~line 346) and the K-fold Mermaid in
  `01-data-science/03-worked-examples/10-feature-selection.md` (~line 197).

## Claims to ground (Haiku — light/none)
- [ ] If AdaBoost's exact weight-update or gradient boosting's residual step is written as a formula,
      confirm it against an authoritative source (sklearn user guide / the original refs) + date.
      Otherwise no new external claims — this is mechanism exposition of already-grounded concepts.

## Acceptance criteria
- [ ] AC1 — bagging and boosting each get a step-by-step Mermaid diagram + a hand-checkable numeric
      walk-through, added to §4; the sequential-vs-parallel contrast is visually explicit.
- [ ] AC2 — existing content unchanged; any added snippet compiles (`check_snippets.py`).
- [ ] AC3 — any new formula grounded (NOTE or inline citation + date).
- [ ] AC4 — renders on GitHub (`check_markdown_render.py` pass; diagrams/formulas eyeballed).

## Gates
Exit: ACs satisfied; fresh-Sonnet review; architect merge. (See `docs/definition-of-done.md`.)
