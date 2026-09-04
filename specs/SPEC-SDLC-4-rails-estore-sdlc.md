# SPEC-SDLC-4: Governing an AI-built Rails e-store — user login and checkout under gates

**Status:** approved
**Subject:** AI-assisted-sdlc
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-SDLC-1 (Theory — prompts, rules, hooks, gates, tools, sub-agents, skills),
SPEC-SDLC-2 (Scaffolding a governed SDLC for a Java project). Sibling to SDLC-2 on a different stack.

## Intent
SDLC-2 scaffolds a governed AI-assisted pipeline for a *Java* project and drives one small, pure-logic
feature (a Luhn validator) through it. This chapter raises the stakes on purpose: a **different stack
(Ruby on Rails)** and a **security-sensitive web app** — a small e-commerce store where the two
features the reader drives through the loop are **user authentication (sign-up / login / logout)** and
**checkout (cart → order → payment seam)**. Those are exactly the features where an unsupervised AI
edit does real damage — a leaked secret, a missing authorization check, a mass-assignment hole, an
unhashed password — so they're the perfect vehicle for showing that the **governance layer (rules,
gates, sub-agents) is what makes AI-assisted development safe**, not the model's good intentions.

The lesson is the *workflow*, not Rails trivia: the reader learns to encode project rules
(`CLAUDE.md`), stand up an implementer/reviewer/researcher sub-agent split, wire **security-first
gates** (RuboCop for style, RSpec for tests, **Brakeman** for a static security scan, a secret-scan
guard), write a feature spec, and watch the spec → implement → review → gate → merge loop produce
real, reviewed code — with the reviewer catching a deliberately-planted security slip on the way.

## Learning objectives
After this chapter the reader can:
- LO1 — Scaffold a governed SDLC for a **non-Java** stack: a Rails project's `CLAUDE.md` rules,
  `.claude/agents/` (implementer, reviewer, researcher), `.claude/hooks/` and `.claude/settings.json`,
  and a `docs/definition-of-done.md` tuned for a web app (authorization, no secrets, tests, security scan).
- LO2 — Write a **feature spec** for an app feature (not just a pure function): user login, then
  checkout — each with acceptance criteria a reviewer can check.
- LO3 — Wire **security-first gates** for AI-generated web code and explain *why each exists*: RuboCop
  (style/lint), RSpec/Minitest (tests), **Brakeman** (mass-assignment, SQL-injection, unsafe redirects),
  bcrypt password hashing via `has_secure_password`, strong parameters, CSRF, and a guard that blocks a
  secret (a Stripe key, `Rails.application.credentials`) from being printed or committed.
- LO4 — Run the **feature loop** twice: (a) **login** — sign-up, session create/destroy, `has_secure_password`,
  authorize-before-action; (b) **checkout** — cart → `Order`/`LineItem`, an idempotent order create, and
  a **PaymentService seam** (Stripe shown as the production integration in **test mode**, keys from ENV/
  credentials, the real charge call stubbed so it runs with no account and no secret in the repo). Show
  the reviewer requesting a change (a planted authorization or mass-assignment slip) and the fix.
- LO5 — Explain what transfers back to any stack and any app, and where a human must stay in the loop
  (payment correctness, authorization, data deletion) — honest about the app being *correct for a
  declared Rails environment*, not executed inside this Python book repo.

## Scope
In scope: a small but real Rails app (User/Session, Product, Cart/LineItem, Order) with sign-up/login/
logout and a cart→checkout→order flow with a stubbed payment seam; the full governance scaffold; two
feature specs driven through the loop; security-first gates; two loop transcripts (one showing a
caught defect). Code is **idiomatic, correct, and runnable in the declared Rails environment**; it is
NOT executed in this repo (no Ruby toolchain here) — state that plainly.
Out of scope (name + link): real payment processing / live Stripe keys (prohibited and unnecessary —
use a stubbed seam); deployment (link SDLC/agentic cloud chapters); a full storefront UI beyond the
minimal views needed to exercise login + checkout; admin. No real secret ever appears in the repo.

## Assets to produce
- Prose: `04-ai-assisted-sdlc/03-worked-examples/03-rails-estore-sdlc.md`
- Project: `04-ai-assisted-sdlc/03-worked-examples/code/rails-estore/` mirroring the Java example's
  shape, adapted to Rails:
  - `.claude/agents/{implementer,reviewer,researcher}.md`, `.claude/hooks/{guard,verify,context}.sh`,
    `.claude/settings.json`
  - `CLAUDE.md` (Rails project rules), `.rubocop.yml`, `Gemfile` (Rails + rspec-rails + brakeman +
    rubocop, pinned per grounding), `config/routes.rb`
  - `docs/architecture.md`, `docs/definition-of-done.md`,
    `docs/features/FEATURE-1-user-login.md`, `docs/features/FEATURE-2-checkout.md`
  - `app/models/` (user.rb with `has_secure_password`, product.rb, cart.rb, line_item.rb, order.rb),
    `app/controllers/` (sessions, registrations, products, carts/line_items, checkout/orders),
    minimal `app/views/`, `app/services/payment_service.rb` (the stubbed Stripe seam)
  - `spec/` (or `test/`) — model + request/controller specs for login and checkout, including the
    security cases (unauthorized access blocked, password hashed, mass-assignment blocked)
  - `.env.example` (documents `STRIPE_SECRET_KEY` etc. — never a real value)
- Artefacts (mirror the Java ones): `artefacts/rails-feature-loop-transcript.md` (login + checkout,
  one showing a CHANGES-REQUESTED → fix), `artefacts/rails-validation-log.md` (real-shaped RuboCop/
  RSpec/Brakeman output), and a feature-loop Mermaid (reuse the pattern; a PNG is optional).

## Claims to ground (Haiku research brief — do BEFORE writing)
- [ ] Current stable **Ruby** and **Rails** versions (PyPI-equivalent: rubygems.org / official releases,
      with dates). Confirm **Rails 8's built-in authentication generator** (`bin/rails generate
      authentication`) and its shape (has_secure_password, Session model, Current, cookies) — is this
      the current idiomatic auth, and how does it compare to Devise? Recommend the dependency-light
      native route unless grounding says otherwise.
- [ ] Current versions of the gate gems: `rspec-rails`, `rubocop` (+ `rubocop-rails`), `brakeman`,
      `bcrypt`. Confirm Brakeman is the standard Rails static security scanner and name 2–3 real check
      categories it reports (mass assignment, SQL injection, unsafe redirect/render).
- [ ] The idiomatic **checkout/order** modelling in current Rails (Order + LineItem, order state), and
      the current **Stripe** integration seam in test mode (the gem, `Stripe::Checkout::Session` or
      PaymentIntent, keys via `Rails.application.credentials`/ENV) — enough to write a faithful *stub*
      with a real integration point, citing Stripe/Rails docs. NO real keys.
- [ ] Confirm `has_secure_password` uses bcrypt and the strong-parameters / mass-assignment protection
      story, cited to the Rails guides.

## Acceptance criteria (each maps to evidence)
- [ ] AC1 (LO1–LO2) — the governance scaffold + two feature specs exist and mirror the Java example's
      structure, adapted to Rails → evidence: the file tree.
- [ ] AC2 (LO3) — security-first gates are wired and each is justified by a real failure it catches;
      the guard blocks a secret; Brakeman/RuboCop/RSpec are in `verify.sh` → evidence: the hooks + the
      validation-log artefact.
- [ ] AC3 (LO4) — login and checkout are implemented as idiomatic, correct Rails (has_secure_password,
      authorize-before-action, Order/LineItem, stubbed PaymentService), with tests including the
      security cases; one loop transcript shows a reviewer catching a planted slip and the fix →
      evidence: the app code + specs + transcripts.
- [ ] AC4 — no real secret anywhere; `.env.example` documents config; payment is a stub with a real
      integration seam → evidence: a grep-clean check noted in the chapter.
- [ ] AC5 — honest environment note: code is correct/runnable in the declared Rails env, not executed
      in this Python repo; every external claim (versions, auth generator, Stripe/Brakeman) grounded.
- [ ] AC6 — renders on GitHub (`check_markdown_render.py` pass on the chapter + artefacts; Mermaid
      labels with parens quoted); the chapter's own `.md` passes the render gate.
- [ ] AC7 — repository coherence: `docs/curriculum.md` (architect), the AI-assisted-SDLC `README.md`
      worked-examples list, and cross-links to SDLC-1/SDLC-2 updated.

## Gates
Entry: this spec approved; research NOTEs landed. Exit: all ACs satisfied; the chapter's markdown
renders; internal links resolve; fresh-Sonnet review sign-off; architect merge.
(See `docs/definition-of-done.md`.)
