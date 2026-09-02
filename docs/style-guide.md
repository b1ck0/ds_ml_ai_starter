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
