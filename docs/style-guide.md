# Style guide — writing for a senior Java engineer

The reader is one specific person: a backend engineer with 15+ years in Java, strong on systems,
types, testing, and build tooling — but new to Python, data science, and ML. Write every chapter for
**that** reader.

## Voice
- Direct and concrete. Lead with the "why" (the problem it solves), then the "how".
- Assume deep programming maturity; do **not** assume any ML/stats background.
- One idea per section. Short paragraphs. Prefer a runnable example over three paragraphs of theory.

## Use the reader's existing mental models
Reach for Java/JVM analogies **when they genuinely clarify**, then drop them once the Python idea
stands on its own. Examples of the kind of bridge that helps:

- `pip` / `venv` / `requirements.txt` ≈ Maven/Gradle + dependency scopes; a virtualenv ≈ an isolated
  classpath per project.
- NumPy arrays / pandas DataFrames ≈ columnar, vectorised collections — think "streams over primitive
  arrays, but the loop is in C", not `ArrayList<Object>`.
- A scikit-learn `Pipeline` ≈ a composable chain of transforms with one `fit`/`predict` contract —
  like a well-typed builder with a single interface.
- Jupyter ≈ a REPL/worksheet with persisted cell state — closer to a Java worksheet than to a `main`.

Do not force an analogy that misleads (e.g. Python typing is not Java typing). When it would mislead,
say plainly "this is unlike Java here, and why".

## Show the artefact
Every worked example ends in something the reader can see: a plot, a metrics table, a confusion
matrix, a printed DataFrame. Commit the artefact under the chapter's `artefacts/` and reference it.

## Code conventions
- Python 3.12+. Snippets are complete and runnable: real imports, pinned deps, no `...` elisions
  unless the block is explicitly pseudocode.
- Prefer small, deterministic examples (set random seeds). Reproducibility is a feature.
- State the environment the snippet expects (which `requirements` / extras).

## Grounding conventions
- Never state a version, a dataset URL, a metric formula, or "library X does Y" from memory. It comes
  from a `research/NOTE-*.md` (Haiku-verified) or an inline citation with the date checked.
- Cite inline as: `[source: <short name>](<url>) (checked 2026-09-02)`, and/or reference the NOTE id.

## Structure of a chapter
1. **What & why** — the problem, and where a Java engineer would meet it.
2. **Concept** — the minimum theory, grounded.
3. **Worked example** — dataset → code → artefact → interpretation.
4. **Pitfalls** — what goes wrong, how to see it.
5. **Recap & what's next** — link to the next chapter in the curriculum.

## Storytelling & visualisation (house style)
The book is taught as a **story**, one idea at a time, with a picture on nearly every beat. Model the
"Data Science for non-IT" deck: a curious reader with no ML background should be pulled through it.

**Storytelling moves — use these deliberately:**
- **Cold open with a concrete story, not theory.** Start each chapter (or major section) with a real,
  relatable case, then name the concept it illustrates. (E.g. regression opens with Prof. Ashenfelter
  predicting Bordeaux wine prices, `price = 0.642·AGST − 3.547`, before the word "regression".) Reduce
  the idea to one plain sentence the reader could repeat at dinner.
- **Numbered steps, one idea each.** Break a concept into "Step 1 … Step 2 …", each a single move.
- **Build with real numbers first, then generalise.** Show the actual arithmetic on a tiny example
  (`0.642×17 − 3.547 = 7.37`), *then* write the general formula. Never open with the general formula.
- **Error-driven narrative for iterative work.** Frame feature engineering / model improvement as a
  journey: "model error 50% → add a distance feature → 30% → bucket the map → 20% → …", one change per beat.
- **Recurring "you are here" map.** Keep one process/architecture diagram for the subject and re-show
  it (with the current step highlighted) at each section boundary, so the reader never loses the thread.
- **Historical timelines** where a field evolved (e.g. CNNs: LeNet → AlexNet → VGG → ResNet →
  EfficientNet), each entry naming *what it introduced* and *why*.

**Visualise constantly — prefer GitHub-native, theme-aware formats (no external tooling):**
- **Mermaid** for flow, sequence, timeline, mindmap, state, and block/architecture diagrams — fenced as
  ```` ```mermaid ````. GitHub renders it natively. Aim for **at least one diagram per major section**;
  a wall of prose with no picture is a smell. Keep diagrams small and labelled; avoid colour that breaks
  in dark mode (rely on shapes/labels, not colour alone).
- **LaTeX math** via `$…$` / `$$…$$` — GitHub renders it. Use it for every formula (metrics, losses,
  update rules) instead of ASCII.
- **Tables** for comparisons ("which model favours numeric vs categorical", "cloud service mapping").
- **The runnable artefacts you already generate** (plots, confusion matrices, parity plots) stay — the
  Mermaid/story wraps around them, it doesn't replace them.

**Reusing the owner's slide material (`docs/curriculum.md` lists the decks):**
- You MAY reuse the owner's own explanations and **original diagrams** (redrawn as Mermaid/SVG is
  preferred — cleaner, theme-aware, versionable). Recreate rather than screenshot wherever practical.
- Do **NOT** commit third-party images embedded in those decks (dataset photos, ImageNet/Ranger-7
  imagery, logos, book/paper figures) into this public repo — recreate the idea as a diagram instead.
- When you do embed an owner-original image, put it under the chapter's `artefacts/` and attribute it.

## Tailored for people (accessibility)
Write so a smart person with **no IT/maths background** could follow — the owner's "for non-IT" and
Galton "regression to the mean" decks are the bar. Concretely:
- **Plain-language intuition BEFORE notation or code.** Say what it *means* in one human sentence, then
  show the formula, then the code. Never open a concept with a formula.
- **Every symbol gets a plain gloss** the first time (`$\mu$` = "the average", `SSE` = "how wrong the
  model is", `SST` = "how wrong the dumbest model is").
- **Analogies over jargon.** Compare to something the reader already lives (a courtroom, a pricing
  function, a flaky test), then retire the analogy once the real idea stands on its own.

## Explanatory patterns worth reusing (from the owner's decks)
Two set-piece explanations to reuse (recreated as runnable plots + Mermaid/LaTeX, not slide screenshots):
- **R² as "beat the dumbest model."** Show two residual pictures side by side: residuals to the fitted
  line ($SSE=\sum_i (y_i-\hat y_i)^2$, "variance of the model's errors") vs residuals to the flat
  baseline $\hat y=\bar y$ ($SST=\sum_i (y_i-\bar y)^2=\text{VAR}(Y)$, "variance of the data"). Then
  $R^2 = 1-\frac{SSE}{SST}$ = "the fraction of the data's variance the model explains": $R^2=0$ when the
  model is no better than predicting the average, $R^2=1$ when it's perfect. Generate the two-residual
  plot with real code as an artefact.
- **Collinearity via "multiple simple regressions vs one multiple regression."** Rank each feature by
  its OWN single-feature $R^2$ (many simple regressions). Then fit ONE multiple regression on
  **standardised** features so coefficient *magnitude* ≈ importance, and compare the coefficient ordering
  to that single-feature ranking. When features are collinear (e.g. `Age = base − Year`, a population
  proxy that also tracks `Year` — "three features secretly measuring time"), the multiple-regression
  weights go unstable and the ordering breaks (a redundant twin collapses to ≈0; the design matrix's
  **condition number**/VIF explodes). Drop the redundant twins → the coefficient ordering snaps back to
  the single-feature importance ranking. Show the pairplot, the coefficient-vs-single-R² comparison, and
  the condition number, all from runnable code.
