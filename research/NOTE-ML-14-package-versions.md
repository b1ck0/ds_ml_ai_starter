# NOTE-ML-14: Current stable PyPI versions of NLP metric libraries

**Date checked:** 2026-09-03

## Answer
Pin these versions for reproducibility:
- `sacrebleu==2.6.0` (2026-01-12)
- `rouge-score==0.1.2` (2022-07-22)
- `bert-score==0.3.13` (2023-02-20)
- `scikit-learn==1.9.0` (2026-06-02)
- `evaluate==0.4.6` (2025-09-18)
- `torch==2.14.0` (2026-09-02)
- `transformers==5.16.1` (2026-08-26)

## Evidence

**sacrebleu 2.6.0** — Released 2026-01-12
- Source: [WebSearch result](https://pypi.org/project/sacrebleu/)
- Verified against PyPI search results

**rouge-score 0.1.2** — Released 2022-07-22
- Source: [PyPI page](https://pypi.org/project/rouge-score/)
- Described as "Pure python implementation of ROUGE-1.5.5"

**bert-score 0.3.13** — Released 2023-02-20
- Source: [PyPI page](https://pypi.org/project/bert-score/)
- Release note: "Fix bug with transformers version > 4.17.0"

**scikit-learn 1.9.0** — Released 2026-06-02
- Source: [PyPI page](https://pypi.org/project/scikit-learn/)
- Requires Python 3.11 or newer

**evaluate 0.4.6** — Released 2025-09-18
- Source: [PyPI page](https://pypi.org/project/evaluate/)

**torch 2.14.0** — Released 2026-09-02
- Source: [PyPI page](https://pypi.org/project/torch/)
- Includes wheel distributions for Python 3.10–3.14

**transformers 5.16.1** — Released 2026-08-26
- Source: [PyPI page](https://pypi.org/project/transformers/)
- Described as "model-definition framework for state-of-the-art ML"

## Caveats & limits
- **rouge-score** is old (2022) and unmaintained; no newer version available on PyPI. The `rouge_score` library is a pure Python implementation of ROUGE-1.5.5; for production use, verify it aligns with your ROUGE definition needs.
- **torch and transformers** move quickly; versions beyond 2026-09 may exist. Lock to these exact versions in `requirements.txt` to avoid runtime surprises.
- **evaluate** version 0.4.6 (Sept 2025) is recent; newer dev versions may exist in GitHub.

## Recommendation
- **Pin all seven versions exactly** in a `requirements.txt` or `pyproject.toml` for reproducibility across runs.
- Test the chapter's code snippets against these specific versions before merging; metric APIs and defaults can shift subtly between major/minor versions.
- For the chapter example, explicitly state the version constraints in comments (e.g., `# sacrebleu 2.6.0+` or `# tested with torch 2.14.0`).
