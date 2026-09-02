---
name: research-brief
description: Write a scoped research brief for the Haiku researcher and turn its findings into a research/NOTE-*.md. Use before any chapter asserts an external fact — a current package version, a dataset URL and licence, a library/API's real behaviour, or a metric's exact definition.
---

# research-brief

No chapter should assert an external fact from memory. When a chapter spec lists "claims to ground",
the architect writes a tight brief, dispatches the **Haiku researcher**
(`.claude/agents/researcher.md`), and the researcher returns a `research/NOTE-<n>-<slug>.md`.

## The brief must contain
- **Question(s):** the specific facts to verify, phrased so the answer is checkable — "what is the
  current stable version of pandas on PyPI, and on what date?", not "look into pandas".
- **Method:** where to look (PyPI for versions, the official docs for APIs, the dataset's canonical
  page for the URL + licence), and to record the source URL and the date checked.
- **Deliverable:** `research/NOTE-<n>-<slug>.md` with: the answer, evidence (quoted versions /
  signatures / URLs + date), caveats/limits, and a clear recommendation for the writer.
- **Boundaries:** free/authorized sources only; never commit secrets; if it can't be verified, say so
  rather than guessing.

## Why this matters here
A wrong version number or a dead dataset link silently breaks every snippet the writer builds on top.
Grounding is the "test suite" of a textbook: the researcher records the truth once, and every chapter
that cites the NOTE inherits it.

## NOTE-*.md shape
```markdown
# NOTE-<n>: <question>
**Answer:** <the verified finding, one line>
**Evidence:** <source URLs, quoted versions/signatures/fields, date checked>
**Caveats / limits:** <ambiguity, conflicting sources, version-sensitivity>
**Recommendation:** <how the chapter writer should use this>
```
