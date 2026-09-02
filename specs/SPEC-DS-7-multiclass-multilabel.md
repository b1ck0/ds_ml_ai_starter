# SPEC-DS-7: Multi-class and multi-label classification

**Status:** approved
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-6

## Intent
Extend binary classification to (a) pick one of N classes and (b) attach several labels at once.
Clarify the often-confused distinction and how metrics/averaging change.

## Learning objectives
- LO1 — Distinguish multi-class (exactly one of N) from multi-label (any subset of N), with concrete examples.
- LO2 — Train a multi-class classifier and read a multi-class confusion matrix; understand one-vs-rest vs softmax.
- LO3 — Understand metric averaging: micro / macro / weighted, and when each is right.
- LO4 — Train a multi-label classifier (`MultiOutputClassifier` / native) and evaluate with per-label + averaged metrics (Hamming loss, subset accuracy).

## Scope
In: multi-class (one dataset) + multi-label (one dataset), averaging strategies, the relevant metrics.
Out: extreme multi-label / hierarchical classification (mention + link), deep learning (→ ML subject).

## Outline
1. What & why — three shapes of classification target; a Java `enum` vs a `Set<Enum>`.
2. Multi-class — dataset, softmax vs OvR, confusion matrix, macro/micro/weighted F1.
3. Multi-label — dataset, binary-relevance approach, per-label report, Hamming loss & subset accuracy.
4. Pitfalls — treating multi-label as multi-class; averaging that hides a weak class.

## Assets to produce
- Prose: "Data Science/Worked Examples/multiclass-multilabel.md"
- Code: "Data Science/Worked Examples/code/multiclass_multilabel.py"
- Artefacts: multi-class confusion matrix heatmap; per-label metric table.

## Claims to ground (Haiku, before writing)
- [ ] Pick a good MULTI-CLASS dataset with a verified loader (e.g. sklearn `load_digits` / `load_wine` / `load_iris`, or seaborn) — confirm shape.
- [ ] Pick a good MULTI-LABEL dataset with a verified source/loader (e.g. sklearn `make_multilabel_classification` for a fully-runnable synthetic option, or a real one like "yeast"/"emotions" if easily loadable) — recommend the runnable path.
- [ ] Reuse NOTE-5; confirm APIs: OneVsRestClassifier, MultiOutputClassifier, f1_score(average=...), hamming_loss, accuracy_score for subset accuracy.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — code runs (both tasks) + artefacts + snippet-check. AC3 — datasets + averaging + multilabel metric APIs grounded. AC4 — the enum-vs-set framing used; averaging pitfalls shown.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.
