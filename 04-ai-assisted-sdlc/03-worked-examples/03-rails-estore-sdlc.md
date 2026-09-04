# Governing an AI-Built Rails E-Store — User Login and Checkout Under Gates

*AI-assisted-sdlc · Worked Examples · SPEC-SDLC-4 + SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents*

> **The point of this chapter is not to hand you a finished Rails store to read — it's for you to
> build one.** Everything narrated below happened when a governed loop, driven by a human steering
> Claude Code, built this app feature by feature; §6a is the step-by-step version of that same loop
> with your hands on the keyboard, and
> [`code/rails-estore/README.md`](code/rails-estore/README.md#build-it-yourself-with-claude-code) is
> the same walkthrough written to run standalone, without this chapter open. The committed
> `rails-estore/` code is the **destination** — where you end up after driving the loop yourself —
> not the point of reading this chapter.

## The bug that only requires being logged in as anyone

Ask an agent, with no gates in place, to "let a signed-in user view their past orders," and here is
the fastest path from prompt to a green checkmark: add a route, add an action, look the order up by
the id in the URL, render it. It works the first time you click it — your own order shows up, because
you're testing with your own account. The PR looks done: a controller action, a view, a passing
manual click-through.

```ruby
# app/controllers/checkout/orders_controller.rb — what an ungated agent ships
def show
  @order = Order.find(params[:id])
end
```

Nothing about this line is malformed Ruby, unstyled, or a SQL-injection risk — it will sail through a
linter and a security scanner without a single warning, for reasons this chapter gets to directly.
What it actually does is let *any signed-in user* read *any other user's order* by changing one digit
in the URL: sign in as anyone, visit `/checkout/orders/1`, `/checkout/orders/2`, `/checkout/orders/3`,
and read every customer's order history — names, products, order totals — one increment at a time.
That is not a hypothetical: it is the single most common access-control bug in web applications,
formal enough to have its own name, IDOR (insecure direct object reference), and its own line item in
the [OWASP Top 10](https://owasp.org/Top10/A01_2021-Broken_Access_Control/) (A01:2021 — Broken Access
Control) (checked 2026-09-04). A registration form one `permit()` call away from letting a new
signup set `admin: true` on themselves is the same failure mode wearing a different outfit: the code
runs, the tests (if any exist) are testing the happy path, and the defect is invisible until someone
specifically goes looking for it — usually an attacker, not a code reviewer skimming a diff.

[SPEC-SDLC-2 (the Java scaffold)](01-java-sdlc-scaffold.md) made this same argument for a
`-DskipTests` shortcut on a pure-logic Luhn validator: a hook, not good intentions, is what makes the
shortcut *not run at all*. This chapter raises the stakes on purpose — a different stack (Ruby on
Rails 8.1) and a genuinely security-sensitive app, a small e-store with real login and real checkout
— because login and checkout are exactly the features where an unsupervised AI edit does real
damage, and exactly where "the governance layer makes AI-assisted development safe, not the model's
good intentions" stops being an abstract claim and becomes something you can watch happen, twice,
below.

## 1. What & why — the same four primitives, a higher-stakes stack

[SPEC-SDLC-1 (Theory)](../01-theory/01-theory.md) named four scaffolding primitives an agentic
coding tool needs: **prompts & rules** (`CLAUDE.md` — advisory, read by the architect),
**hooks & gates** (deterministic automation that can actually *block* a bad action),
**tools & MCP** (typed capabilities, here `Bash` running `bundle exec rspec`/`rubocop`/`brakeman`),
and **sub-agents & skills** (fresh-context dispatch: researcher, implementer, fresh reviewer).
[SPEC-SDLC-2](01-java-sdlc-scaffold.md) proved all four port to a Java/Maven codebase unchanged in
shape. This chapter proves the same four port to Ruby/Rails — and adds one thing neither prior
chapter needed: **the gate's content has a threat model.** A textbook chapter's gate asks "does this
compile and render." A Luhn validator's gate asks "does this compile, and did the test fail first."
This app's gate asks those *and* "can a user read another user's data" and "can a user set an
attribute nobody exposed to them" — because this is the first stack in this book with a login form
and a payment integration behind it.

Everything below lives under [`code/rails-estore/`](code/rails-estore/) — a complete Rails project
scaffold with its own `.claude/` governance layer nested inside it, exactly as it would sit in a real
repository. §2–3 tour the original three-role scaffold; §3a introduces two more specialist reviewers
this addendum adds — `seo-optimizer` and `frontend-qa`; §4–5 cover the hooks and gates. §6 runs all
three features — login, then checkout, then the product catalog — through the full loop, including
the moment a fresh reviewer catches a real vulnerability that three green automated gates missed, and
the moment the two new specialists each catch a defect in their own lane that those same gates don't
even attempt to check. §6a hands you the keyboard: the same loop, condensed into a repeatable
checklist with paste-ready prompts, so you drive FEATURE-1/2/3 yourself instead of only reading how
they went. §7 lists what breaks the loop if you skip a step. §8 is the direct comparison:
this repo's own scaffold, `java-project/`, and `rails-estore/`, side by side. §9 is honest about what
actually ran where, and points to [`code/rails-estore/README.md`](code/rails-estore/README.md) for
running the whole thing for real on a Mac.

**Toolchain baseline**, grounded in
[`research/NOTE-SDLC-4-1-versions.md`](../../research/NOTE-SDLC-4-1-versions.md) (checked
2026-09-04): Ruby **4.0.6** ([Ruby 4.0.6 released](https://www.ruby-lang.org/en/news/2026/07/14/ruby-4-0-6-released/),
July 14, 2026), Rails **8.1.3.1** ([Rails 8.0.5 and 8.1.3 released](https://rubyonrails.org/2026/3/24/Rails-Versions-8-0-5-and-8-1-3-have-been-released),
patched to `.1` July 29, 2026), `rspec-rails` **8.0.4**, `rubocop` **1.90.0** + `rubocop-rails`
**2.37.0** (bumped up from the grounding note's `1.86.0` — a real `bundle install`, run for real in
Docker per SPEC-SDLC-4-ADDENDUM-2, found `rubocop-rails 2.37.0` now requires `rubocop >= 1.89.0`; see
`artefacts/rails-validation-log.md` §7), `brakeman` **8.0.6**, `bcrypt` **3.1.22** (includes a
CVE-2026-33306 fix), `stripe` **19.3.0**. Every one of these is pinned by exact version in
[`Gemfile`](code/rails-estore/Gemfile). **This sandbox has no Ruby/Rails toolchain** — §9's
Environment note is explicit about which evidence below is real, captured output, and which is a
grounded reference command you run on a machine that has Ruby and Rails installed. The one exception
is Docker itself, which **is** available here — the docker-compose path (§9, "Running it") was
actually built, booted, and tested end-to-end, not just described.

### The map

```mermaid
flowchart TB
    subgraph RULES["Prompts and rules -- advisory, read by the architect"]
        CLAUDEMD["CLAUDE.md"]
        ARCHMD["docs/architecture.md"]
        DODMD["docs/definition-of-done.md"]
    end
    subgraph ROSTER["Sub-agents and skills -- fresh-context dispatch"]
        RESEARCHERMD[".claude/agents/researcher.md -- Haiku"]
        IMPLEMENTERMD[".claude/agents/implementer.md -- Sonnet"]
        REVIEWERMD[".claude/agents/reviewer.md -- fresh Sonnet"]
        SEOMD[".claude/agents/seo-optimizer.md -- Sonnet"]
        FRONTENDQAMD[".claude/agents/frontend-qa.md -- Sonnet"]
    end
    subgraph GATEBOX["Hooks and gates -- the only layer that can block"]
        GUARDSH[".claude/hooks/guard.sh -- PreToolUse on Bash"]
        VERIFYSH[".claude/hooks/verify.sh -- PostToolUse on Edit/Write"]
        CONTEXTSH[".claude/hooks/context.sh -- SessionStart"]
    end
    subgraph WIRING["Tools and MCP -- the typed capability an agent calls"]
        SETTINGSJSON[".claude/settings.json"]
    end
    RULES -.->|"architect (Opus) reads first"| ROSTER
    SETTINGSJSON -->|"wires hook scripts to lifecycle events"| GATEBOX
    ROSTER -->|"Bash calls the implementer makes"| GATEBOX
```

Identical shape to [SPEC-SDLC-2's map](01-java-sdlc-scaffold.md#the-map-four-primitives-one-file-each)
— because the settings schema and the agent frontmatter schema genuinely don't know or care what
language the gate checks. §4 is where that stops being true: the *content* of `guard.sh` and
`verify.sh` is where a Rails app's threat model actually shows up.

## 2. The charter + docs — rules, with a security golden rule Java didn't need

[`code/rails-estore/CLAUDE.md`](code/rails-estore/CLAUDE.md) mirrors this repository's own
`CLAUDE.md` and `java-project/CLAUDE.md`'s seven/eight-rule shape, with one rule neither prior
project's charter needed to state:

| This repo (content authoring) | `java-project/` (Java feature) | `rails-estore/` (Rails feature) |
|---|---|---|
| No chapter without an approved `specs/SPEC-*.md` | No code without an approved `docs/features/FEATURE-*.md` | Same |
| Every claim grounded (NOTE or citation) | Every dependency version/API claim grounded | Same, plus: every security/CVE claim |
| Every snippet must run | Every JUnit 5 test must FAIL before the code that satisfies it | Every RSpec example must FAIL before the code that satisfies it |
| *(no equivalent — no auth surface)* | *(no equivalent)* | **Golden rule 4: every `Current.user`-scoped query is authorized; every `permit()` list is minimal** — this repo and `java-project/` have no login or payment surface for this rule to apply to |
| Exit gate: snippet compile + link check + review | Exit gate: `mvn test` + checkstyle + spotless + spotbugs + review | Exit gate: `rspec` + rubocop + `brakeman -q --no-summary` (zero High-confidence warnings) + review |

[`docs/architecture.md`](code/rails-estore/docs/architecture.md) names the same six sections as
both prior scaffolds (operating model, roles, workflow, repository shape, gates, decision log) — its
decision log's second-newest entry records exactly why the Rails 8 *native* auth generator was
chosen over Devise: dependency-light and officially maintained as of Rails 8, sufficient for this
app's scope
[source: NOTE-SDLC-4-2-auth-generator.md]. [`docs/definition-of-done.md`](code/rails-estore/docs/definition-of-done.md)
adds a "Security" section between "Grounded" and "Green gate" that doesn't exist in either prior
project's checklist — authorization, mass assignment, password hashing, and the secret scan are each
their own checkbox with their own required RSpec evidence, not folded into "correctness."

## 3. The agent roster — same three roles, one process addition on the reviewer

[`.claude/agents/researcher.md`](code/rails-estore/.claude/agents/researcher.md) (Haiku) and
[`.claude/agents/implementer.md`](code/rails-estore/.claude/agents/implementer.md) (Sonnet) are
structurally identical to `java-project/`'s — verify a fact/write a note; write a failing test, then
the minimum code, then run the gate. The implementer's process adds one line the Java implementer
didn't need: *"For anything touching authentication, authorization, or mass-assignable params,
default to the MORE restrictive option and let the spec's acceptance criteria justify opening it up
— never the other way round."*

[`.claude/agents/reviewer.md`](code/rails-estore/.claude/agents/reviewer.md) (a fresh Sonnet) is
where the real addition lives. Where `java-project/reviewer.md`'s process is fidelity → test-first →
grounded → green gate → scope, this project's reviewer inserts two explicit, named steps in between:

> 3. **Authorization, by hand, on every controller action touched.** ... confirm the query is
>    actually scoped to the current user (`Current.user.orders.find(...)`, not `Order.find(...)`) —
>    an unscoped `find` on an id from the URL is an IDOR even if every other check passes.
> 4. **Mass assignment, by hand, on every controller action that builds or updates a model from
>    params.** Confirm `permit()` lists ONLY the attributes the feature actually needs, by name.

*Why we do it this way:* a linter and a security scanner both have a fixed catalog of patterns they
check for. Reading `Order.find(params[:id])` and asking "whose order is this allowed to be" is a
judgment call about what the *feature* is supposed to authorize, not a pattern match against a known
CWE — which is exactly why §6's checkout loop shows this line passing RuboCop and Brakeman *both
clean*, and why the reviewer's instructions say "by hand" twice rather than once.

## 3a. Two more specialists — SEO and frontend quality (this addendum)

FEATURE-1 and FEATURE-2 gave this store an account system and a checkout, both genuinely
security-sensitive. But neither the general `reviewer.md` above nor RuboCop/Brakeman has any opinion
on two other dimensions an e-store cannot ship without: **can a search engine (or a shared link)
find and correctly represent this page**, and **can a person using a keyboard or a screen reader
actually use it**. Those are different failure modes from an IDOR — not a missing authorization
check, a missing *category of check entirely*. RuboCop's rule catalog has no "form input needs a
label" rule; Brakeman's 86 checks (§5) cover injection and mass-assignment classes of bug, never
WCAG or schema.org. So the roster grows by two, each earning its place the same way `reviewer.md`
did: by catching something real, in its own lane, that nothing else in this project's gate catches.

| Specialist | Checks | Real tooling | Catches what security/style gates don't |
|---|---|---|---|
| [`seo-optimizer.md`](code/rails-estore/.claude/agents/seo-optimizer.md) | Unique title/description, canonical, Open Graph, schema.org **Product** JSON-LD, `robots.txt`/sitemap, heading hierarchy | `meta-tags` 2.24.0, `sitemap_generator` 7.1.1, `@lhci/cli` 0.15.1 (reference) | A missing/invalid Product JSON-LD block — the page is syntactically fine, renders correctly, and is simply invisible to a merchant-listing search result |
| [`frontend-qa.md`](code/rails-estore/.claude/agents/frontend-qa.md) | WCAG/axe (labels, alt, contrast, heading order, `lang`, focus/skip link, landmarks), valid/semantic/responsive HTML, broken links | `axe-core-rspec`/`axe-core-capybara` 4.13.0, `html-proofer` 5.2.2 | An unlabelled form field or a broken heading order — code that is stylistically perfect Ruby and a stylistically perfect view, unusable by a screen-reader user |

Both are structured exactly like `reviewer.md`: a numbered checklist, "output a verdict, do not
merge." Both are wired the same three places every role in this project is wired:
[`CLAUDE.md`](code/rails-estore/CLAUDE.md)'s Model routing and Gates sections name them (a feature
that adds or changes a rendered page additionally clears the *frontend gate*, in addition to
`rspec`/`rubocop`/`brakeman`), and
[`docs/definition-of-done.md`](code/rails-estore/docs/definition-of-done.md) gets a new **SEO &
Accessibility** section — parallel in shape to the existing "Security" section (§5's authorization/
mass-assignment/password-hashing checklist), with its own required evidence per checkbox rather than
folded into "correctness." `frontend-qa.md`'s own instructions carry the same caveat
[NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md](../../research/NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md) states
plainly: automated axe checks catch roughly 30–40% of real WCAG issues, so a clean axe run is a
floor for this checklist, never the whole verdict.

*Why we do it this way:* the same argument §3's authorization/mass-assignment addition made for
security applies here with the words swapped. A linter has a fixed catalog of patterns; "is this
input's label programmatically associated with it" and "is this JSON-LD block valid per Google's
Merchant Listing requirements" are judgment calls about what the *page* needs to do, not pattern
matches against a known rule — which is exactly why both new agents' checklists say "by hand" and
"re-run it yourself" rather than "trust the implementer's report," the same discipline §3's reviewer
process already established. §6's FEATURE-3 walkthrough shows both specialists catching a real,
planted defect each — one in a form field, one in a missing structured-data block — that a fully
green `rspec`/`rubocop`/`brakeman` run reported nothing about.

## 4. Hooks + settings — the gate's content is where the stack actually matters

[`.claude/settings.json`](code/rails-estore/.claude/settings.json) is byte-for-byte the same three
events and handler shape as every prior scaffold — validated for real, not just eyeballed, in
[`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §1. The three hook scripts
carry the stack-specific content:

- **[`guard.sh`](code/rails-estore/.claude/hooks/guard.sh)** (`PreToolUse` on `Bash`) keeps every
  deny-rule from this repo's own `guard.sh` (`rm -rf /`, forced push, printing a secret) and adds
  two rules a Java content-authoring project has no use for: it blocks `git commit --no-verify`
  (the Rails-project equivalent of the Java chapter's `-DskipTests` — a way to talk the gate out of
  running at all), and — the flagship new rule — **it blocks any shell command containing a
  live-looking Stripe key** (`sk_live_`/`pk_live_`/`rk_live_`). This is what turns "no real secret
  ever appears in the repo" from a written rule in `CLAUDE.md` into something enforced at the shell
  level, before an agent can even echo a key while debugging, let alone commit one. Real, captured
  proof it fires — five cases, all five correct — is in
  [`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §3.
- **[`verify.sh`](code/rails-estore/.claude/hooks/verify.sh)** (`PostToolUse` on `Edit|Write`) runs
  `bundle exec rspec`, then `rubocop`, then `brakeman -q --no-summary` on any `.rb` edit — the three
  tools §5 grounds as this stack's standard quality/security gate — and `bundle check` on a `Gemfile`
  edit, the same "validate the manifest immediately" pattern `java-project/verify.sh` runs for
  `pom.xml`.
- **[`context.sh`](code/rails-estore/.claude/hooks/context.sh)** (`SessionStart`) prints the same
  orientation banner pattern as both prior projects — read-first docs, the roster, and every
  `docs/features/FEATURE-*.md` with its status. Real output in
  [`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §3.

## 5. Security-first gates, and why each one exists

Four gates, four different failure modes, each named because a real one caught something in this
chapter's own worked example:

**RuboCop** (style/lint) — catches nothing security-specific by default, but `.claude/hooks/verify.sh`
runs it first because it's the fastest signal: a controller action too long to read in one glance
(`Metrics/MethodLength`, capped at 15 lines in
[`.rubocop.yml`](code/rails-estore/.rubocop.yml)) is a controller action too long for a reviewer to
authorize-check by eye, which is exactly the check that matters most on this app.

**RSpec** (`rspec-rails` 8.0.4) — the test-first discipline from `java-project/` ported unchanged:
an example is written and confirmed failing before the code that satisfies it. What's new here is
*what* the examples cover: `spec/requests/registrations_spec.rb` and `spec/requests/checkout_spec.rb`
each carry a security-specific example (mass assignment, IDOR) alongside the functional ones — see
§6.

**Brakeman** (8.0.6) — the standard Rails static security scanner, 86 checks covering SQL injection,
mass assignment, XSS, CSRF, unsafe redirects, and more
[source: NOTE-SDLC-4-3-brakeman-checks.md]. Three worth naming because this chapter's own code
exercises them:
- **Mass assignment** — flags a model built from unpermitted params
  [source: [Brakeman — Mass Assignment](https://brakemanscanner.org/docs/warning_types/mass_assignment/)
  (checked 2026-09-04)]. `RegistrationsController#user_params`'s three-key `permit()` list is why
  this app reports clean.
- **SQL injection** — flags unescaped user input in a raw query (e.g.
  `User.where("email = '#{email}'")`); this app avoids the pattern entirely by using
  ActiveRecord's parameterized finders (`find_by(email_address: ...)`) everywhere.
- **Cross-Site Scripting** — flags unescaped output, most commonly a stray `.html_safe`.
  [`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §4 walks a real example
  of this exact check catching a real early draft of this chapter's own `products/show.html.erb`.

*Why we do it this way:* Brakeman is static analysis — it never runs the app. Its own documentation
is explicit about the boundary that matters most for this chapter: it "will not catch logic errors
(e.g., authorization gaps that require dynamic analysis or test coverage)"
[source: NOTE-SDLC-4-3-brakeman-checks.md, "Caveats"]. §6's checkout loop is built entirely around
that one sentence.

**bcrypt + `has_secure_password`** — the User model's `has_secure_password` call hashes with bcrypt,
a deliberately slow algorithm (~200–300 ms per hash) chosen specifically to make brute-forcing a
stolen password-hash dump impractical
[source: [ActiveModel::SecurePassword — Rails API](https://api.rubyonrails.org/classes/ActiveModel/SecurePassword/ClassMethods.html)
(checked 2026-09-04)]. The `password_digest` column stores only the hash; `spec/models/user_spec.rb`
asserts the digest never equals the plaintext, directly — see §6.

**Strong parameters** — Rails' mass-assignment defence since Rails 4: a controller must explicitly
`permit()` every attribute it will assign from user input, or that attribute is silently dropped
[source: [Rails Strong Parameters Deep Dive](https://blog.saeloun.com/2025/02/18/deep-dive-into-rails-action-controller-strong-parameters/)
(checked 2026-09-04)]. Every controller in `app/controllers/` that builds or updates a model from
`params` uses it; `spec/requests/registrations_spec.rb` proves the `admin` param specifically cannot
be set this way.

**CSRF** — Rails enables `protect_from_forgery` by default for every `ActionController::Base`
subclass (which `ApplicationController` is); this app takes the default rather than opting out of it
anywhere, so it is not called out as a separate feature — it is the one gate here that required no
code to get right, only *not disabling it*.

## 6. The loop in action

```mermaid
flowchart LR
    SPEC["spec: docs/features/FEATURE-*.md<br/>written by the architect (Opus)"] --> GROUND{"new external<br/>fact needed?"}
    GROUND -->|"yes"| RESEARCH["researcher.md (Haiku)<br/>grounds it, writes it down"]
    GROUND -->|"no -- already landed"| IMPL
    RESEARCH --> IMPL["implementer.md (Sonnet)<br/>failing RSpec example first, then the code"]
    IMPL --> GATEBOX2["gate: guard.sh + verify.sh<br/>rspec + rubocop + brakeman -q<br/>(+ axe + html-proofer on view edits)"]
    GATEBOX2 -->|"red"| IMPL
    GATEBOX2 -->|"green"| REVIEW["reviewer.md (fresh Sonnet)<br/>re-runs the gate AND checks authorization by hand"]
    GATEBOX2 -->|"green, UI-facing"| SEOREVIEW["seo-optimizer.md (Sonnet)<br/>titles, OG, Product JSON-LD, sitemap"]
    GATEBOX2 -->|"green, UI-facing"| QAREVIEW["frontend-qa.md (Sonnet)<br/>axe/WCAG, valid HTML, broken links"]
    REVIEW -->|"changes requested"| IMPL
    SEOREVIEW -->|"changes requested"| IMPL
    QAREVIEW -->|"changes requested"| IMPL
    REVIEW -->|"approve"| MERGE["merge (architect, Opus)"]
    SEOREVIEW -->|"approve"| MERGE
    QAREVIEW -->|"approve"| MERGE
    MERGE -.->|"loop closes -- next feature"| SPEC
```

`seo-optimizer.md` and `frontend-qa.md` run **in parallel** with the general reviewer, not instead of
it, and only for features that add or change a rendered page — FEATURE-1's sign-in form and
FEATURE-2's checkout flow predate this addendum and were reviewed by `reviewer.md` alone; FEATURE-3
(§3a, below) is the first feature all three review together.

The full stage-by-stage narration for all three features is
[`artefacts/rails-feature-loop-transcript.md`](artefacts/rails-feature-loop-transcript.md), explicitly
labelled a **reference transcript** — real file names, real spec, real committed code, real review
reasoning; illustrative `bundle exec rspec`/`rubocop`/`brakeman`/axe/html-proofer console bytes,
because this sandbox has no Ruby toolchain and no browser either (§9). Here is the shape of what it
shows.

### FEATURE-1 — user login (clean pass)

[`docs/features/FEATURE-1-user-login.md`](code/rails-estore/docs/features/FEATURE-1-user-login.md)
sets six acceptance criteria. The implementer writes
[`spec/models/user_spec.rb`](code/rails-estore/spec/models/user_spec.rb),
[`spec/requests/registrations_spec.rb`](code/rails-estore/spec/requests/registrations_spec.rb), and
[`spec/requests/sessions_spec.rb`](code/rails-estore/spec/requests/sessions_spec.rb) against stub
controllers first, confirms all eight examples fail for the right reason (the behaviour doesn't
exist yet), then writes the real
[`User`](code/rails-estore/app/models/user.rb)/[`Session`](code/rails-estore/app/models/session.rb)/
[`Current`](code/rails-estore/app/models/current.rb)/
[`Authentication`](code/rails-estore/app/controllers/concerns/authentication.rb) and the two
controllers. The gate goes green on the first pass; the fresh reviewer's authorization check finds
nothing to flag — `CartsController#show` never takes a `params[:id]` at all, the simplest form
authorization can take — and mass assignment is a clean three-key `permit()` list. **APPROVE**,
straight line from spec to merge, same shape as `java-project/`'s FEATURE-1.

### FEATURE-2 — checkout (the planted slip, caught and fixed)

[`docs/features/FEATURE-2-checkout.md`](code/rails-estore/docs/features/FEATURE-2-checkout.md) sets
six acceptance criteria, one of them (AC5) naming the authorization case explicitly: *"not satisfied
by 'the route requires sign-in' alone."* The implementer writes
[`spec/models/order_spec.rb`](code/rails-estore/spec/models/order_spec.rb) and
[`spec/requests/checkout_spec.rb`](code/rails-estore/spec/requests/checkout_spec.rb), confirms the
failures are real, then writes
[`Checkout::OrdersController`](code/rails-estore/app/controllers/checkout/orders_controller.rb).
Under time pressure, `show` ships as `Order.find(params[:id])` — this chapter's cold-open bug, for
real — and AC5's RSpec coverage ships as one positive case only ("returns the order to its own
owner"), missing the negative case the spec's own wording asked for.

The gate runs and reports **all three green**: RSpec is green because nothing exercises the
vulnerable path; RuboCop is green because the line is perfectly idiomatic Ruby; Brakeman is green
because broken access control across records of the same type isn't one of its static checks
(§5). The implementer reports the gate green, honestly believing the feature is done.

The fresh reviewer's process step 3 — **"by hand, on every controller action touched"** — finds it
reading the diff directly, in about the time it takes to read the line:

> **CHANGES REQUESTED.**
> 1. `app/controllers/checkout/orders_controller.rb:26` — `Order.find(params[:id])` is not scoped to
>    `Current.user`. Any signed-in user can read any other user's order by incrementing the id in the
>    URL (IDOR, OWASP A01:2021). Fix: `Current.user.orders.find(params[:id])`.
> 2. `spec/requests/checkout_spec.rb` — AC5's negative case has no RSpec example. Green RSpec is not
>    evidence of anything here.

The implementer adds the missing negative example, confirms it fails against the still-broken
controller (a real `ActiveRecord::RecordNotFound` expectation that raises nothing — the concrete
proof the bug is real, not theoretical), applies the one-line fix, and re-runs the gate — this time
with the negative case actually contributing to a green RSpec run. The reviewer re-checks both
findings, re-runs the gate independently, and returns **APPROVE**. Architect merges.

*Why we do it this way:* this is the chapter's whole argument, concentrated into one diff. Three
automated gates were not lying — each was reporting exactly what it is built to report. None of them
is built to answer "is this the right user's data." A fresh reviewer instructed to ask that question
by hand, on every action, every time, is.

### FEATURE-3 — product catalog (two specialist defects, one per new agent)

[`docs/features/FEATURE-3-product-catalog-seo-a11y.md`](code/rails-estore/docs/features/FEATURE-3-product-catalog-seo-a11y.md)
(this addendum) sets nine acceptance criteria for a public, SEO-complete, accessible product catalog
— the first feature `seo-optimizer.md` and `frontend-qa.md` review. The implementer writes
[`spec/system/accessibility_spec.rb`](code/rails-estore/spec/system/accessibility_spec.rb) and
[`spec/system/seo_spec.rb`](code/rails-estore/spec/system/seo_spec.rb), confirms both fail for the
right reason, then writes the real
[`ProductsController`](code/rails-estore/app/controllers/products_controller.rb),
[`ProductsHelper`](code/rails-estore/app/helpers/products_helper.rb), and — new to this project —
[`app/views/layouts/application.html.erb`](code/rails-estore/app/views/layouts/application.html.erb),
which did not exist before this addendum. Two slips ship, one per specialist's lane: the catalog's
search field ships as a bare `<input type="search" name="q" placeholder="Search products">` with no
`<label>` — placeholder text is not a label, and disappears the moment a user starts typing — and the
product show page ships with **no** `<script type="application/ld+json">` block at all.

`bundle exec rspec`, `rubocop`, and `brakeman -q --no-summary` all report clean, for the same reason
§6's IDOR case did: none of the three has a check in this category at all — not a blind spot in an
otherwise-broad tool, a category none of them was ever built to cover. The implementer reports every
check it could run as green.

Dispatched in parallel, `frontend-qa.md` runs the axe system spec for real and finds the missing
label immediately (axe rule `label`, impact **critical**, WCAG 1.3.1/4.1.2) — **CHANGES REQUESTED**.
`seo-optimizer.md` parses the rendered show page and finds no Product JSON-LD anywhere on it — Google
Merchant Listing requires `name`/`image`/`offers.price`/`offers.priceCurrency` at minimum
[NOTE-SDLC-4-ADD-2-schema-product.md](../../research/NOTE-SDLC-4-ADD-2-schema-product.md) —
**CHANGES REQUESTED**. The implementer adds the missing `<%= f.label %>` and the two JSON-LD
`<script>` tags (Product + BreadcrumbList), re-runs the gate, and both specialists — plus the general
`reviewer.md`, which had nothing to flag since this controller touches no `Current.user`-scoped query
or `permit()` call — return **APPROVE**. Architect merges. Full transcript, including both verdicts
verbatim and the illustrative axe/html-proofer output:
[`artefacts/rails-feature-loop-transcript.md`](artefacts/rails-feature-loop-transcript.md) Part C;
reference axe/html-proofer/Lighthouse CI output:
[`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §6.

## 6a. Build it yourself — the same loop, with your hands on the keyboard

§6 narrated what happened when this loop ran. This section is the same six-step rhythm, written as
something you actually do: install Claude Code, open `code/rails-estore/`, and drive
FEATURE-1 → FEATURE-2 → FEATURE-3 yourself. Every step below, with its paste-ready prompt, the
install/sign-in instructions, and the `docker compose up` finish line, lives in
[`code/rails-estore/README.md`](code/rails-estore/README.md#build-it-yourself-with-claude-code) so it
runs standalone, without this chapter open. Here is the shape of it.

```mermaid
flowchart LR
    YOU["you read the spec"] --> PROMPT["you prompt Claude Code"]
    PROMPT --> IMPL["implementer sub-agent<br/>failing RSpec first, then code"]
    IMPL --> GATE["verify.sh + guard.sh fire"]
    GATE -->|"red"| IMPL
    GATE -->|"green"| REVIEW["reviewer -- + seo-optimizer,<br/>frontend-qa for UI -- by hand"]
    REVIEW -->|"changes requested"| IMPL
    REVIEW -->|"approve"| DECIDE["you review the diff, merge"]
    DECIDE -.->|"repeat, next feature"| YOU
```

**The loop, once per feature:**

1. **Read the spec.** [`docs/features/FEATURE-1-user-login.md`](code/rails-estore/docs/features/FEATURE-1-user-login.md),
   then `FEATURE-2-checkout.md`, then `FEATURE-3-product-catalog-seo-a11y.md` — the acceptance
   criteria are the contract you're about to hold the implementer to, not a suggestion.
2. **Prompt Claude Code to implement it.** A starting point:

   > Implement FEATURE-1 (`docs/features/FEATURE-1-user-login.md`). Use the implementer sub-agent:
   > write the failing RSpec examples for every acceptance criterion first, confirm each fails for the
   > right reason, then write the code that makes them pass. Run the full gate before telling me it's
   > done.

   Claude Code dispatches `implementer.md` — by natural-language inference, or a forced
   `@agent-implementer` mention — which writes the failing test before a single line of `app/` code
   [source: [Claude Code — Sub-agents](https://code.claude.com/docs/en/sub-agents) (checked
   2026-09-04)].
3. **Watch the gates fire.** `verify.sh` runs `rspec`/`rubocop`/`brakeman` on every `.rb` edit (§4);
   `guard.sh` vetoes dangerous shell before it runs at all. A red gate on the first pass — a fresh
   RSpec file with every example failing — is correct, not broken; that's the test-first discipline
   §2's golden rule 2 names, made visible in your own terminal instead of read about.
4. **Get an independent review.**

   > Use the reviewer sub-agent to review the diff against the spec's acceptance criteria. Check
   > authorization and mass assignment by hand, not just the gate output.

   This is the step §6's FEATURE-2 narration exists to justify: three green gates and a real IDOR,
   caught only because the reviewer read `Order.find(params[:id])` and asked whose order that query
   is actually scoped to. For FEATURE-3, add `seo-optimizer` and `frontend-qa` to the same prompt —
   they're the roles that catch the unlabelled search field and the missing Product JSON-LD block §6
   walks through, neither of which `rspec`/`rubocop`/`brakeman` has a check for at all.
5. **Repeat for FEATURE-2, then FEATURE-3.** Same rhythm; only the acceptance criteria — and which
   defect class they guard against — change.
6. **Run what you built.** `docker compose up`, then <http://localhost:3000> — the storefront §9's
   screenshot shows.

**How to steer well:** be specific and point at the spec rather than describing the feature from
memory; let the gates and reviewers do the catching instead of hand-checking every RuboCop rule
yourself; review every diff before you call it merged — an **APPROVE** from `reviewer.md` is a strong
signal, not a merge button.

**Honest notes:** model outputs vary run to run — your own FEATURE-2 attempt might ship the IDOR §6
describes, or it might not; either way, the reviewer step is what makes the loop safe, not a promise
that the first draft is already correct. You stay in the loop at every decision point — approve,
reject, merge — which is exactly what makes the *hands-off* stretches (the implementer's edit-test-fix
cycle inside step 2, the gate firing automatically inside step 3) safe to let run without narrating
every keystroke yourself.

## 7. Pitfalls

- **Trusting a green security scanner to mean "no vulnerabilities."** Brakeman reporting zero
  warnings means zero warnings *in the categories Brakeman checks* — §6's IDOR is the concrete
  counter-example. Read `docs/research/NOTE-SDLC-4-3-brakeman-checks.md`'s own "Caveats" section
  before treating any static scanner's silence as a verdict.
- **A permit list that grows by convenience.** `permit(:email_address, :password,
  :password_confirmation, :admin)` compiles, passes RuboCop, and Brakeman only flags mass assignment
  when a permit call is *missing* — a permit call that exists but is too generous reads as
  intentional. The reviewer's step 4 exists because nothing automated distinguishes "this attribute
  belongs here" from "this attribute happens to be here."
- **Testing the happy path and calling it done.** FEATURE-2's implementer wrote a real, passing test
  for AC5 — just the wrong half of it. A green suite is only as trustworthy as the acceptance
  criteria it actually encodes; "AC5 has a test" and "AC5 is tested" are not the same claim.
- **Skipping the failing-test-first step.** Same risk `java-project/`'s Pitfalls named: write the
  test and the code in the same pass, and you've verified they agree with each other, not that the
  test would have caught the bug it claims to guard against.
- **Storing money as a float.** `Product#price_cents` and `Order#total_cents` are integers on
  purpose — a `decimal`/`float` price accumulates rounding error across enough line items and
  discounts to matter; storing cents as an integer makes that class of bug structurally impossible.
- **Assuming "RSpec/RuboCop/Brakeman all green" means the page is done.** FEATURE-3 (§3a, §6) is the
  concrete counter-example for a second class of bug, alongside §6's IDOR: none of those three tools
  has ANY check for a missing form label or a missing structured-data block — not a partial-coverage
  gap, a category outside their remit entirely. `seo-optimizer.md`/`frontend-qa.md` exist because
  "the gate is green" and "this page is finished" are, again, not the same claim.
- **A stub that quietly becomes the real call.** `PaymentService`'s test-mode branch
  (`Rails.env.test? || ENV["STRIPE_STUB"] == "true"`) must stay the FIRST branch checked — the
  reviewer's step 8 explicitly flags "a stubbed `PaymentService` accidentally calling the real Stripe
  client" as a scope-creep pattern to watch for, because it's the one bug in this app that would cost
  real money to trigger.

## 8. Reference — three scaffolds, one shape

| Layer | This repo (content) | `java-project/` (Java) | `rails-estore/` (Rails) | Carries over? |
|---|---|---|---|---|
| `.claude/settings.json` schema | 3 events | identical | identical | **Unchanged** |
| Agent frontmatter schema | `name`/`description`/`model`/`tools` | identical | identical | **Unchanged** |
| `guard.sh` deny-list | `rm -rf /`, forced push, secrets | + `mvn -DskipTests` | + `git commit --no-verify`, + live Stripe key pattern | **Extended per stack's actual risk** |
| `verify.sh` gate | byte-compile a snippet | `mvn test` + checkstyle + spotless + spotbugs | `rspec` + rubocop + `brakeman -q` | **Stack-specific — the one place the language matters most** |
| Spec unit | one chapter | one feature (`FEATURE-*.md`) | one feature (`FEATURE-*.md`) | Same shape |
| "Runnable" gate criterion | snippet compiles | test existed and failed first | RSpec example existed and failed first | Same shape |
| Reviewer's extra checklist item | — | test-first commit order | test-first order **+ authorization by hand + mass-assignment by hand** | **New per stack's threat model** |
| Roster | researcher/writer/reviewer/architect | researcher/implementer/reviewer/architect | identical roles | **Unchanged** |
| Model routing | Haiku grounds, Sonnet writes, fresh Sonnet reviews, Opus scopes+merges | identical | identical | **Unchanged** |

The takeaway, sharper this time than in either prior chapter: everything Claude Code itself provides
— the settings schema, the frontmatter schema, the permission model, the hook lifecycle — is
stack-agnostic and ports verbatim, twice proven now. What changes, every time, is what you author
*for* your stack: the gate's actual tool invocations, and — new to this chapter — the specific,
named questions a human-directed reviewer must ask by hand because no available automated tool asks
them. On a content-authoring repo, that extra question doesn't exist. On a Java feature with no
network-facing auth surface, it doesn't either. On a Rails app that stores passwords and talks to a
payment gateway, it is the single most important check in the whole loop.

*Roster footnote (this addendum):* the table above compares the three scaffolds' **shape**, which is
still unchanged — `rails-estore/`'s roster grows from four roles to six (architect, researcher,
implementer, reviewer, `seo-optimizer`, `frontend-qa`) without touching the settings schema, the
frontmatter schema, or the model-routing pattern any of those roles are wired through. §3a covers the
two additions in full; neither `java-project/` nor this repo's own scaffold has a UI-facing surface
for either role to apply to.

## 9. Environment note, secret scan, and honesty about what ran where

**This code is correct and idiomatic Ruby/Rails 8.1, written to run in a declared Rails environment
— it was not executed inside this Python book repository**, which has no Ruby toolchain (`which
bundle`, `which rspec`, `which brakeman` all resolve to nothing here) **and no browser or Node
toolchain either** (`which chromedriver`, `which node` also resolve to nothing) — so the frontend
gate this addendum adds (axe, html-proofer, `npx lhci autorun`) is reference-only for the same reason
the Ruby gate is. What WAS actually run, for real, in this repository's own sandbox:
`python -m json.tool` against `.claude/settings.json`; `validate_frontmatter.py` against all **five**
agent files (researcher, implementer, reviewer, `seo-optimizer`, `frontend-qa`); all three hook
scripts (`context.sh`/`guard.sh`/`verify.sh`) fed real synthetic Claude Code hook payloads on stdin,
with real captured output for every case, including all five `guard.sh` deny-rules; and a real `grep`
secret scan across the entire `code/rails-estore/` tree — and, added by SPEC-SDLC-4-ADDENDUM-2, a
real `docker compose build`/`up`/`curl`/`bundle exec rspec`/`down -v` cycle, since Docker (unlike
Ruby) genuinely runs in this sandbox. Every one of those is in
[`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) with its exact command and
output, §§1, 2, 3, 5, and 7 marked real, §4 (RSpec/RuboCop/Brakeman) and §6 (axe/html-proofer/
Lighthouse CI) marked as grounded reference reproductions. The feature-loop transcript follows the
same convention — see its own header.

**Running it.** Everything above is about what ran inside *this* book's own sandbox. The app now ships
a **verified docker-compose setup** (SPEC-SDLC-4-ADDENDUM-2) — `docker compose up`, then
<http://localhost:3000> — genuinely built, booted, and tested in this same sandbox (Docker, unlike
Ruby, actually runs here): real `docker compose build`/`up`/`curl`/`bundle exec rspec` output is in
[`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §7, including three real
boot-time bugs the run surfaced and fixed (an rspec-rails API break, `RAILS_ENV` silently resolving
to the wrong environment, two missing stock Rails `test.rb` defaults) and one genuine rendering bug a
screenshot of the running container caught that no test did. If you'd rather run Ruby directly on a
real Mac — sign up, add a product to a cart, check out, and run every gate including the frontend one
— [`code/rails-estore/README.md`](code/rails-estore/README.md) is a complete, standalone,
zero-to-running guide: Homebrew → rbenv → Ruby 4.0.6 → Rails 8.1.3.1, `bin/rails db:setup` (seeded
with four sample products), `bin/rails server`, and every gate (`rspec`/`rubocop`/`brakeman`, plus
the frontend gate's `axe`/`html-proofer`/`npx lhci autorun`), each with a one-line "what it's for" and
a troubleshooting section for the usual native-gem/Apple-Silicon snags. Either path is written to be
read on its own — a friend cloning just `code/rails-estore/` doesn't need this chapter to get the app
running, only to understand *why* it's built the way it is.

Here is that running store — the seeded catalog, captured live from the container during the verified
`docker compose up` run:

![The rails-estore product catalog running under docker-compose: a "Shop all products" heading, a search box, and four seeded products — Rails Mug $15.00, Convention Over Configuration T-Shirt $25.00, Omakase Sticker Pack $8.00, and Migration Notebook $12.00.](code/rails-estore/docs/screenshots/storefront.png)

*It ships no CSS on purpose — this is a governance teaching scaffold, where clean, semantic,
accessible HTML is the point (what the `frontend-qa` and `seo-optimizer` agents check). Styling it is
a natural first task once a reader sets up Claude Code and starts driving the loop themselves.*

**Secret scan, required before this chapter could be reported done:**
`grep -rniE 'sk_live|pk_live|rk_live' code/rails-estore/` and a second pass matching any
`[sp]k_(live|test)_` key-shaped string — every hit is either prose describing the guard rule, or one
of the two `pk_test_XXXX...`/`sk_test_XXXX...` placeholders in `.env.example`, built from literal
`X` characters. No real or real-looking key exists anywhere in this tree. Full output:
[`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §5.

An earlier version of this chapter named a gap here: `code/rails-estore/` shipped the governed
scaffold plus every feature-specific file, but not the surrounding boot files a fresh `rails new`
generates. SPEC-SDLC-4-ADDENDUM-2 closed that gap — `config/application.rb`, `config/boot.rb`,
`config/environment.rb`, the three `config/environments/*.rb` files, `config/puma.rb`, `config.ru`,
`Rakefile`, `bin/rails`, and `bin/setup` are all now committed, minimal, and — unlike everything else
in this chapter's sandbox — genuinely **booted**: `docker compose build && docker compose up` serves
the real app at `http://localhost:3000`, `bundle exec rspec` runs the real suite in the container
(17/17 model + request specs pass), and `docker compose down -v` tears it down clean. The one honest
gap that remains is scoped, not hidden: the 5 browser-driven system specs
(`spec/system/accessibility_spec.rb`, `spec/system/seo_spec.rb`) need a real Chrome + `chromedriver`,
which this minimal, Node-free Docker image deliberately doesn't install — run those via the native
macOS path in [`code/rails-estore/README.md`](code/rails-estore/README.md) §3 instead. Full real
output, including four boot-time bugs the Docker run surfaced and fixed along the way:
[`artefacts/rails-validation-log.md`](artefacts/rails-validation-log.md) §7.

## 10. Recap & what's next

Read the last two sections again with one word changed: not "*the reviewer* caught it," but "*you*,
having dispatched the reviewer, watched it get caught." That is the whole reframe this chapter has
been building toward — §6a (and the standalone [README](code/rails-estore/README.md)) turn every
catch narrated below into something reproducible, keystroke for keystroke, the next time you open
`code/rails-estore/` in Claude Code yourself.

The cold open's bug — `Order.find(params[:id])`, any signed-in user reading anyone's order — got
written twice in this chapter: once as an ungoverned agent's actual first attempt at FEATURE-2 (§6),
and once as this chapter's opening hook, deliberately, before you knew a governed loop was about to
catch it. That's the concrete difference a fresh reviewer instructed to check authorization *by
hand, every time* makes, proven against a real diff, not asserted in the abstract. FEATURE-3 (§3a,
§6) proves the same shape of argument twice more, on two dimensions a security reviewer was never
meant to cover: `frontend-qa.md` caught an unlabelled search field axe was built specifically to
catch; `seo-optimizer.md` caught a missing Product JSON-LD block Google's own merchant-listing
requirements name explicitly. Three specialist findings, three different categories of defect
(authorization, accessibility, discoverability), one shared cause: an automated gate that is green
*in every category it checks* is not the same claim as "this feature is done," and the fix in every
case was the same — a named role, with a checklist, instructed to check by hand what nothing
automated in this project checks at all.

Four primitive categories, one governed loop, now proven three times over four features: a textbook
chapter ([SPEC-SDLC-1](../01-theory/01-theory.md)), a Java feature
([SPEC-SDLC-2](01-java-sdlc-scaffold.md)), and — this chapter — a security-sensitive, now also
SEO/accessibility-governed, Rails feature set. What ported unchanged: the settings schema, the agent
frontmatter schema, the roster *shape* (fresh-context specialists dispatched by the architect,
grounded first, gated always), the model routing. What didn't: the gate's actual content, and the
specific, named judgment calls a human-directed reviewer has to make *by hand* because no available
automated tool makes them for you — first for authorization (§6), now for accessibility and SEO
(§3a, §6's FEATURE-3). Brakeman is real and it is good at what it does; it does not do everything,
and knowing exactly where that line falls (`docs/research/NOTE-SDLC-4-3-brakeman-checks.md`'s own
words: it "will not catch logic errors ... that require dynamic analysis or test coverage") is what
makes every one of these specialist roles non-optional rather than a formality.

If you haven't read them yet, [SPEC-SDLC-1 (Theory)](../01-theory/01-theory.md) is the abstract
version of everything this chapter just proved concretely, and
[SPEC-SDLC-2 (the Java scaffold)](01-java-sdlc-scaffold.md) is the sibling worked example — same
loop, a stack with no login form to get wrong. [**How this repo was built**](02-how-this-repo-was-built.md)
(SPEC-SDLC-3) turns the camera on this very book, tracing the same loop through this repository's
own real commits. If you're setting up a governed loop on your own Rails project right now,
`code/rails-estore/` is a complete, ready-to-adapt starting point — install Claude Code (§6a / the
README's "Set up Claude Code" section), copy `.claude/`, `CLAUDE.md`, and `docs/`, point `Gemfile` at
whatever your own researcher grounds as current when you do it (the
versions pinned here were current 2026-09-04 — check again), replace `FEATURE-1`/`FEATURE-2`/
`FEATURE-3` with your project's actual first features, and run the loop for real, with real `bundle`
on `PATH` — or, to just get the app running on your own Mac first,
[`code/rails-estore/README.md`](code/rails-estore/README.md) is the standalone path there.
