# SPEC-SDLC-4-ADDENDUM: add SEO-optimization and frontend-QA agents to the Rails e-store scaffold

**Status:** in-review
**Subject:** AI-assisted-sdlc
**Section:** Worked Examples (amends `04-ai-assisted-sdlc/03-worked-examples/03-rails-estore-sdlc.md`
and its `code/rails-estore/` project, SPEC-SDLC-4)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8

## Why this addendum
SDLC-4 stands up a governed Rails e-store with implementer/reviewer/researcher agents and
security-first gates, and its teaching climax is a *specialized* reviewer catching a class of defect
(an IDOR) that the automated gates miss. The owner wants to extend that same idea to the two quality
dimensions an e-commerce storefront can't ship without and that a general reviewer + Brakeman/RuboCop/
RSpec do **not** cover: **discoverability (SEO)** and **frontend quality (accessibility, valid/
responsive HTML, no broken links)**. So the agent team grows by two specialists, each with its own
checklist and its own real tooling, each earning its place by catching something the others can't.

This extends — does not rewrite — the committed chapter and project. Preserve existing files
byte-for-byte except where a change is explicitly required below (adding the agents to `CLAUDE.md`
routing, to `verify.sh`, to the DoD, and to the Gemfile).

## What to add
### Two new sub-agents (`code/rails-estore/.claude/agents/`)
- **`seo-optimizer.md`** — a specialized reviewer/implementer for **on-page SEO**: a unique,
  descriptive `<title>` and meta description per page; canonical URLs; Open Graph / Twitter card tags;
  **structured data** (schema.org **Product** JSON-LD with offers/price/availability on product pages,
  BreadcrumbList); a generated **sitemap.xml** and **robots.txt**; semantic heading hierarchy (one
  `<h1>`); descriptive link text; image `alt` for indexable images. It flags missing/duplicate titles,
  a missing/invalid Product schema, a noindex left on by mistake, and non-canonical duplicate URLs.
- **`frontend-qa.md`** — a specialized reviewer for **frontend quality**: **accessibility (WCAG / axe)**
  — form inputs have associated `<label>`s, images have `alt`, colour-independent state, focus order,
  a skip link, `lang` on `<html>`; **valid, semantic HTML**; **responsive** layout (viewport meta, no
  fixed-width overflow); **no broken internal links**; **no console errors**. It runs axe on the
  rendered pages and an HTML/link check, and flags a real a11y violation (e.g. an unlabelled checkout
  field or a broken heading order).

Both agents follow the existing agents' shape (frontmatter + a numbered checklist + "output a verdict,
do not merge"), and both are added to `CLAUDE.md`'s **Model routing** and **Gates** sections, and to
`docs/definition-of-done.md` (new SEO + accessibility exit criteria for any UI-facing feature).

### Gate tooling (grounded — real gems/tools, wired into `verify.sh` as the frontend gate)
Wire the real tools the two agents drive, and reference them in `rails-validation-log.md`:
- Accessibility: **axe-core-rspec / axe-core-capybara** in a system spec.
- HTML + link validity: **html-proofer** (or w3c_validators).
- SEO/perf audit: **Lighthouse CI** (`@lhci/cli`, npm) as a reference gate (Node tool — reference-only
  here, like the Ruby gates).
- SEO helpers: the **meta-tags** gem (per-page title/description/OG) and **sitemap_generator** gem.

### Expand the example (the vehicle for the two agents)
Add a small **product catalog** surface so there's real UI to optimize and QA:
- `app/controllers/products_controller.rb` (index + show), `app/views/products/{index,show}.html.erb`
  and a proper `app/views/layouts/application.html.erb` — **semantic, accessible, SEO-complete**:
  labelled forms, `alt` text, one `<h1>`, `meta-tags`/`display_meta_tags`, OG tags, canonical, and a
  **Product JSON-LD** block on the show page.
- `public/robots.txt` and a **sitemap** (`config/sitemap.rb` via sitemap_generator, or a
  `SitemapsController` rendering `sitemap.xml`).
- `spec/system/accessibility_spec.rb` (axe) and a spec asserting the Product JSON-LD + unique title.
- A new feature spec `docs/features/FEATURE-3-product-catalog-seo-a11y.md` driven through the loop:
  implementer builds the catalog → **frontend-QA** catches an a11y violation (unlabelled field / bad
  heading order) → fix → **seo-optimizer** catches a missing Product JSON-LD or a duplicate title →
  fix → reviewer + gates → merge.

### Update the chapter prose
Add a section to `03-rails-estore-sdlc.md` introducing the expanded five-agent team (with an updated
Mermaid of the loop showing SEO + frontend-QA as parallel specialist reviewers), why an e-store needs
them, what each catches that the security/general gates don't, and the FEATURE-3 loop where each
catches a real defect. Update any "you are here"/roster references and the recap.

### A polished, standalone macOS local environment setup (extra priority)
This example will be read **in isolation by someone who checks only `code/rails-estore/`**, so its
setup must stand entirely on its own and be genuinely runnable on **macOS** from zero. Deliver:
- A polished **`code/rails-estore/README.md`** — a real project README that a Mac user can follow end
  to end: install prerequisites (Homebrew → `rbenv` (or asdf) → Ruby 4.0.6 → Rails 8.1.3.1; Node LTS
  for Lighthouse; the DB), `bundle install`, copy `.env.example` → `.env`, `bin/rails db:setup` (with
  seeds for a few sample products so the catalog isn't empty), run the app (`bin/rails server`), and
  run **every gate** (`bundle exec rspec` / `rubocop` / `brakeman`; the frontend gate: html-proofer,
  axe system spec, `npx lhci autorun`). Include a short **troubleshooting** subsection for the usual
  macOS snags (Command Line Tools, a native-extension gem needing a `brew`-installed lib, Apple-silicon
  notes). Explain what each gate is *for* in one line, and point back to the book chapter for the
  governance story. Every command correct and copy-pasteable; the DB choice (SQLite for zero-config,
  or Postgres via `brew install`) stated and consistent with the `Gemfile`/`database.yml`.
- Add `config/database.yml`, a `db/seeds.rb` (a handful of products), and a `bin/setup`-style note if
  it helps — whatever makes the standalone macOS run actually work.
- In the **chapter prose**, add a short "Running it on macOS" pointer to that README (the chapter keeps
  the book's honest note that the tools aren't executed inside *this* Python repo; the README is the
  friend's self-contained path on a real Mac).

## Grounding (Haiku — do BEFORE writing)
- [ ] Current versions + exact usage of: **meta-tags**, **sitemap_generator**, **html-proofer**,
      **axe-core-rspec** (and/or axe-core-capybara), and **Lighthouse CI** (`@lhci/cli`). Confirm each
      is current and how it's invoked (a one-liner per tool).
- [ ] The **schema.org Product** structured-data shape for JSON-LD (required/recommended properties:
      name, image, description, sku, brand, offers{price, priceCurrency, availability}), and Google's
      guidance — cite Google Search Central / schema.org.
- [ ] The core **WCAG / a11y** checks axe enforces that matter here (labels, alt, contrast, heading
      order, landmark/lang) — cite the axe / WCAG docs. Enough to write faithful checklist items and a
      realistic axe report.
- [ ] `robots.txt` + `sitemap.xml` conventions and Open Graph required properties (og:title, og:type,
      og:image, og:url) — cite.

## Acceptance criteria
- [ ] AC1 — `seo-optimizer.md` and `frontend-qa.md` exist, mirror the existing agents' shape, and are
      wired into `CLAUDE.md` routing/gates and the DoD → evidence: the files + the diffs.
- [ ] AC2 — the product-catalog surface is added: semantic/accessible/SEO-complete views, Product
      JSON-LD, robots.txt + sitemap, and the corresponding specs (axe + JSON-LD/title) → evidence: the
      file tree.
- [ ] AC3 — the FEATURE-3 loop transcript shows **frontend-QA** catching a real a11y violation and
      **seo-optimizer** catching a real SEO defect, each fixed → evidence: the extended transcript;
      the validation log shows axe/html-proofer/lighthouse (reference) output.
- [ ] AC4 — existing content preserved except the required wiring changes; no real secret; honest that
      the Ruby/Node tools are not executed in this Python repo (reference output labelled as such).
- [ ] AC5 — every tool/version/schema claim grounded (NOTE ids or inline citation + date).
- [ ] AC6 — renders on GitHub (`check_markdown_render.py` on the chapter + artefacts; Mermaid labels
      with parens quoted). No key-shaped string (learned from the push-protection incident): any
      placeholder must NOT be a `pk_/sk_`-prefixed long character run.
- [ ] AC7 — coherence: chapter, `04-ai-assisted-sdlc/README.md`, and the `docs/curriculum.md` SDLC-4
      bullet (architect) updated to mention the five-agent team.
- [ ] AC8 — a polished, **standalone macOS** `code/rails-estore/README.md` exists: zero-to-running on a
      Mac (prereqs, install, DB + seeds, run, every gate, troubleshooting), every command correct and
      grounded (Homebrew/rbenv/Ruby 4.0.6/Rails 8.1.3.1/Node), consistent with the Gemfile + database
      config; a "Running it on macOS" pointer added to the chapter.

## Gates
Exit: ACs satisfied; the chapter + artefacts render; internal links resolve; fresh-Sonnet review;
architect merge. (See `docs/definition-of-done.md`.)
