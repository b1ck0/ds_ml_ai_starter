# rails-estore — Governed SDLC (living document)

Single source of truth for HOW this codebase gets built: the workflow, the roles, the quality gates,
and the repository shape. Keep it current — update it in the same PR that changes the thing it
describes. This file, and the rest of this project's `.claude/` scaffold, is a direct port of
`ds_ml_ai_starter`'s own content-authoring scaffold onto a Ruby/Rails stack — the same port
[`java-project/docs/architecture.md`](../../code/java-project/docs/architecture.md) already did onto
Java/Maven. See `AI-assisted-sdlc/Worked Examples/03-rails-estore-sdlc.md` §7 of the textbook this
project ships as a worked example for.

## 1. Operating model

Spec-driven, grounded, test-first, **security-first** engineering. The owner states a feature
request; the **architect (Opus)** turns it into an approved feature spec with acceptance criteria;
the **researcher (Haiku)** grounds any external unknown first; **Sonnet** implements — a failing
RSpec example before any production code, defaulting to the more restrictive authorization/
mass-assignment choice whenever the spec is silent — until every gate is green; a **fresh Sonnet**
reviews for correctness, test-first discipline, authorization, mass assignment, and scope; the
architect merges. Nothing merges until its exit gate (§5) passes.

## 2. Roles & model routing

| Role | Model | Responsibility |
|---|---|---|
| Architect | Opus (main session) | Scope features, sequence, gate decisions, review, merge, escalation |
| Implementer | Sonnet (`.claude/agents/implementer.md`) | Implement ONE approved feature: failing test → production code → green gate |
| Researcher | Haiku (`.claude/agents/researcher.md`) | Ground external claims (gem versions, API behaviour, CVEs) → a note the implementer cites |
| Reviewer | Sonnet, fresh (`.claude/agents/reviewer.md`) | Independent QA against the spec + Definition of Done before merge, with authorization/mass-assignment as first-class checks |
| SEO optimizer | Sonnet (`.claude/agents/seo-optimizer.md`) | Specialist review of any UI-facing feature: unique title/meta, canonical, Open Graph, Product JSON-LD, robots.txt/sitemap, one `<h1>` |
| Frontend QA | Sonnet (`.claude/agents/frontend-qa.md`) | Specialist review of any UI-facing feature: axe/WCAG accessibility, valid/semantic/responsive HTML, no broken links or console errors |

The last two run only on features that add or change a rendered page (anything under `app/views/`);
each catches a class of defect — a missing Product schema, an unlabelled input — that the security
and general reviewers plus RuboCop/RSpec/Brakeman do not.

## 3. Workflow (per feature)

1. **Request → feature spec.** Architect writes `docs/features/FEATURE-<n>-<slug>.md`: intent,
   acceptance criteria, and any claims that need grounding. Owner/architect approves. No code before
   approval.
2. **Ground the unknowns (if any).** For every external fact the feature will rely on — a gem's
   pinned version, a library's documented behaviour, a known-CVE check — the architect dispatches the
   **researcher**, which returns a grounding note.
3. **Failing test first.** Dispatch the **implementer**. It writes an RSpec example that encodes the
   acceptance criteria, confirms it fails for the *right* reason (the behaviour doesn't exist yet,
   not a typo), then writes the minimum production code to make it pass.
4. **Gate.** `bundle exec rspec`, `bundle exec rubocop`, `bundle exec brakeman -q --no-summary` (zero
   High-confidence warnings) all pass. `.claude/hooks/verify.sh` runs the fast subset automatically
   on every `.rb` edit.
5. **Review.** Dispatch a **fresh reviewer** for independent QA against the acceptance criteria, the
   Definition of Done, and — specifically — authorization scoping and mass-assignment boundaries on
   every controller action the feature touches, not just whether the gate is green.
6. **Merge.** Architect merges; the PR body maps every acceptance criterion to its evidence.

## 4. Repository shape

- `app/models`, `app/controllers`, `app/services`, `app/views` — standard Rails layout.
- `spec/` — RSpec model + request specs, one file per feature area, including the security cases.
- `docs/features/` — one `FEATURE-<n>-<slug>.md` per feature.
- `docs/` — this file + `definition-of-done.md`.
- `.claude/` — `agents/` (researcher, implementer, reviewer, seo-optimizer, frontend-qa), `hooks/` (verify, guard, context),
  `settings.json`.
- `Gemfile`, `.rubocop.yml` — dependencies and the two style/security gate gems' configuration.
- `.env.example` — every config variable, documented, never a real value.

## 5. Gates (Definition of Done)

Every feature must clear this before it is "done":

- **Fidelity:** every acceptance criterion in the approved feature spec is met.
- **Test-first:** an RSpec example existed, and failed for the right reason, before the production
  code that satisfies it.
- **Grounded:** any gem version, library-behaviour claim, or CVE check traces to a researcher's note
  or a live, dated citation.
- **Authorized:** every action reading or writing a `Current.user`-scoped record is scoped to that
  user, not looked up by a bare id from the URL.
- **No mass assignment:** every `permit()` call lists only the attributes the feature needs; no
  privilege/state field (`admin`, `role`, `status`) is ever permitted from user input.
- **Green gate:** `bundle exec rspec` passes; `rubocop` is clean; `brakeman -q --no-summary` reports
  zero High-confidence warnings.
- **No secrets:** nothing resembling a live key (`sk_live_`/`pk_live_`/`rk_live_`) anywhere in the
  diff.
- **Reviewed:** independent fresh-Sonnet review sign-off + architect merge approval.

Full checklist: `docs/definition-of-done.md`. Fast checks automated on edit by
`.claude/hooks/verify.sh`; the secret pattern is additionally blocked at the shell level by
`.claude/hooks/guard.sh`, so it can't even be echoed during a debugging session.

## 6. Decision log

Non-obvious decisions, newest first — so future work doesn't re-litigate them.

- 2026-09-04 — Chose the Rails 8 **native** `bin/rails generate authentication` scaffold over Devise:
  dependency-light, officially maintained as of Rails 8, and sufficient for this app's sign-up/
  login/logout scope. Devise remains the right call for a project needing 2FA, omniauth, or the
  other modules it bundles — explicitly out of scope here. See
  `docs/research/NOTE-SDLC-4-2-auth-generator.md`.
- 2026-09-04 — Modelled `LineItem` with a polymorphic `belongs_to :cartable, polymorphic: true`
  (`Cart` or `Order`) rather than two separate line-item tables, so a checkout is "copy the cart's
  line items onto a new Order," not a schema migration in disguise.
- 2026-09-04 — Scaffolded from `ds_ml_ai_starter`'s content-authoring `.claude/` framework, following
  the same port `java-project/` already did: chapter specs → feature specs, snippet-compile gate →
  `bundle exec rspec` + RuboCop + Brakeman, "chapter" roles → researcher/implementer/reviewer/
  architect (same model routing). New for this stack: authorization and mass-assignment are named,
  first-class review checks, not folded into "correctness" — because on a Java content-authoring
  project there is no login or payment surface for either failure mode to exist on.
