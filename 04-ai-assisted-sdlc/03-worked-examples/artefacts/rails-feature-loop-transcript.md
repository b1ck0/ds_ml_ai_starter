# Reference transcript — FEATURE-1 and FEATURE-2 through the governed loop

> **This is a REFERENCE transcript, not a captured session log.** It narrates, stage by stage, what
> dispatching `rails-estore`'s architect/researcher/implementer/reviewer roster actually produces —
> using the real files this chapter committed under [`code/rails-estore/`](../code/rails-estore/) —
> so you can see the shape of the loop without paying for a live run. There is **no Ruby toolchain in
> this Python book repository** (see [`03-rails-estore-sdlc.md`](../03-rails-estore-sdlc.md)'s
> Environment note), so every `bundle exec rspec` / `rubocop` / `brakeman` console block below is
> **illustrative**: the file names, the spec, the code, and the review verdicts are all real and match
> what's committed; the exact console bytes of a run are not a transcript of an actual execution.

---

## Part A — FEATURE-1: user login (clean pass, first try)

### Stage 1 — Request → spec

**Owner prompt to the architect:**
> "Before there's a cart, there has to be an account. Add sign-up, login, logout, using Rails 8's
> built-in auth generator shape, not Devise."

**Architect (Opus)**, under `CLAUDE.md` golden rule 1, writes
[`docs/features/FEATURE-1-user-login.md`](../code/rails-estore/docs/features/FEATURE-1-user-login.md):
intent, six acceptance criteria (AC1–AC6), and a "claims to ground" section naming three notes the
spec relies on.

### Stage 2 — Ground the unknowns

The three claims the spec cites were already grounded before this feature was scoped:
`docs/research/NOTE-SDLC-4-2-auth-generator.md` (the generator's shape and the Devise comparison),
`docs/research/NOTE-SDLC-4-1-versions.md` (bcrypt version/CVE status), and
`docs/research/NOTE-SDLC-4-5-auth-security.md` (`has_secure_password`'s bcrypt hashing and strong
parameters). No new dispatch needed — this is the same "stage exists, but is a no-op when nothing is
left to verify" pattern the Java chapter's FEATURE-1 also showed.

### Stage 3 — Failing test first, then implement

**Architect dispatches the implementer** (`.claude/agents/implementer.md`, Sonnet) against
`FEATURE-1-user-login.md`.

**Implementer, step 1 — write the RSpec examples before the code exists to satisfy them:**

```
$ implementer writes spec/models/user_spec.rb, spec/requests/registrations_spec.rb,
    spec/requests/sessions_spec.rb
$ implementer writes STUB versions of User, Session, Current, Authentication, and the two
    controllers (models with no validations/has_secure_password yet; controllers that just
    `render plain: "not implemented", status: :not_implemented`)
$ bundle exec rspec spec/models/user_spec.rb spec/requests/registrations_spec.rb spec/requests/sessions_spec.rb
```

```text
[REFERENCE — illustrative RSpec output, not captured in this sandbox]
Failures:

  1) User password hashing (FEATURE-1 AC2) stores a bcrypt digest, never the plaintext password
     NoMethodError: undefined method `authenticate' for #<User:0x00007f2c1a3b0e10>
  2) Registrations POST /registration creates a user and signs them in (FEATURE-1 AC1)
     Failure/Error: expect(response).to redirect_to(root_path)
       expected: a redirect to "http://www.example.com/"
            got: 501 (not implemented)
  3) Registrations POST /registration ignores an unpermitted admin param ...
     Failure/Error: expect(user).to be_present
       expected User(id: nil) to be present, got nil
  ...
8 examples, 8 failures
```

Good — every failure is the RIGHT kind (the behaviour genuinely doesn't exist yet: no
`has_secure_password`, no controller logic), not a typo in the spec file itself. That's the check
`CLAUDE.md` golden rule 2 and `docs/definition-of-done.md`'s "Test-first" section both demand before
a single line of real logic is written.

**Implementer, step 2 — write the minimum production code:** replaces the stubs with the real
`User`/`Session`/`Current`/`Authentication` and the two real controllers — exactly the files
committed under [`code/rails-estore/app/`](../code/rails-estore/app/).

### Stage 4 — Gate

`.claude/hooks/guard.sh` (`PreToolUse`) ran silently under every `Bash` call the implementer made —
it would have blocked, for example, `git commit --no-verify` outright had the implementer tried to
route around a failing gate. `verify.sh` (`PostToolUse`) fired after every `Edit`/`Write` to a `.rb`
file:

```text
[REFERENCE — illustrative; a real run needs bundle on PATH]
[verify] ruby gate: app/models/user.rb
  $ bundle exec rspec
  $ bundle exec rubocop
  $ bundle exec brakeman -q --no-summary
[verify] all three gates green
```

The implementer reports the full `docs/definition-of-done.md` checklist back to the architect,
mapping AC1–AC6 to the RSpec examples that cover them:

| AC | RSpec example |
|---|---|
| AC1 | `Registrations POST /registration creates a user and signs them in` |
| AC2 | `User password hashing stores a bcrypt digest, never the plaintext password` |
| AC3 | `Registrations POST /registration ignores an unpermitted admin param` |
| AC4 | `Sessions POST /session signs in / rejects incorrect credentials` |
| AC5 | `Sessions DELETE /session destroys the current session` |
| AC6 | `Sessions an unauthenticated request to a protected page redirects to sign-in` |

### Stage 5 — Review

**Architect dispatches a FRESH reviewer** (`.claude/agents/reviewer.md`, a new Sonnet context that
never saw the implementer's scratch work). It walks its own process in order: fidelity against each
AC, test-first evidence (the step-1 failure log above), authorization (`carts_controller.rb` — no
`params[:id]` lookup at all, the simplest form of "can't be wrong"), mass assignment
(`registrations_controller.rb#user_params` — exactly three keys, no `:admin`), grounding (all three
cited notes exist and say what the spec claims), a `sk_live`/`pk_live` grep across the diff (none),
then independently re-runs `bundle exec rspec`, `rubocop`, `brakeman -q --no-summary` itself rather
than trusting the implementer's report.

**Reviewer verdict:**
> **APPROVE.** AC1–AC6 each map to a passing RSpec example; the test-first order is evidenced by the
> step-1 failure log; `rubocop`/`brakeman` are both clean; `registrations_controller.rb`'s permit
> list is minimal and does not include `admin`; no file outside `FEATURE-1`'s "Assets to produce"
> changed. No findings.

### Stage 6 — Merge

**Architect merges**, under `CLAUDE.md`'s merge-approval rule. PR body maps every acceptance
criterion to its evidence (the table above + the gate log).

---

## Part B — FEATURE-2: checkout (a planted authorization slip, caught and fixed)

### Stage 1 — Request → spec

**Owner prompt to the architect:**
> "Now that a user can sign in, let them add products to a cart and check out. Stub the actual
> Stripe charge — no account, no real key — but wire the real integration point."

**Architect** writes
[`docs/features/FEATURE-2-checkout.md`](../code/rails-estore/docs/features/FEATURE-2-checkout.md):
six acceptance criteria (AC1–AC6), explicitly calling out AC5 as "the authorization case; it is not
satisfied by 'the route requires sign-in' alone" — a deliberate, spec-level flag for the exact defect
this stage is about to show.

### Stage 2 — Ground the unknowns

`docs/research/NOTE-SDLC-4-4-stripe-checkout.md` (Order/LineItem shape, `Stripe::Checkout::Session`,
test-mode key prefixes) and `docs/research/NOTE-SDLC-4-3-brakeman-checks.md` (Brakeman's check
categories — cited by the reviewer below, not the implementer) were already landed; no new dispatch.

### Stage 3 — Failing test first, then implement (the slip)

**Implementer, step 1:** writes `spec/models/order_spec.rb` and `spec/requests/checkout_spec.rb`,
confirms every example fails for the right reason against stub controllers, same discipline as
Part A. Under time pressure, though, it writes AC5's example as a single POSITIVE case only —
*"returns the order to its own owner"* — and never writes the negative case the spec's own wording
calls for (*"does NOT return another user's order"*). Nothing catches this at write-time: RSpec has
no way to flag a missing example, only a failing one.

**Implementer, step 2:** writes `Checkout::OrdersController`. The `create` action is fine — it
builds the order through `Current.user.orders.create!`. The `show` action ships with the bug:

```ruby
# app/controllers/checkout/orders_controller.rb — AS FIRST WRITTEN (the planted slip)
def show
  @order = Order.find(params[:id])   # <-- not scoped to Current.user
end
```

This is a textbook IDOR (insecure direct object reference): any signed-in user can read any other
user's order by changing the id in the URL. `docs/architecture.md`'s "closed by default" posture
(§1) is about *authentication* — being signed in at all — and this action still requires that; the
bug is that being signed in as *anyone* is treated as authorization to read *anyone's* order.

### Stage 4 — Gate (green — and that's the point)

```text
[REFERENCE — illustrative]
[verify] ruby gate: app/controllers/checkout/orders_controller.rb
  $ bundle exec rspec
  $ bundle exec rubocop
  $ bundle exec brakeman -q --no-summary
[verify] all three gates green
```

All three gates report clean, for a reason worth sitting with rather than a coincidence:

- **RSpec is green** because the missing negative example means nothing exercises the vulnerable
  path — a green suite proves the tests that exist pass, not that the right tests exist.
- **RuboCop is green** because `Order.find(params[:id])` is perfectly idiomatic, well-styled Ruby.
  Style has no opinion on which model a query is scoped to.
- **Brakeman is green** because broken access control across records of the *same* type isn't one
  of its static checks — `docs/research/NOTE-SDLC-4-3-brakeman-checks.md`'s own "Caveats" section
  says this outright: Brakeman "will not catch logic errors (e.g., authorization gaps that require
  dynamic analysis or test coverage)." This is exactly that gap.

The implementer reports the gate green and hands off for review, honestly believing FEATURE-2 is
done — the report isn't dishonest, it's just measuring the wrong thing.

### Stage 5 — Review (CHANGES REQUESTED)

**A fresh reviewer** is dispatched. Its process (`.claude/agents/reviewer.md` step 3) requires
checking authorization **by hand, on every controller action touched** — not inferring it from a
green gate. Reading `Checkout::OrdersController#show` directly, it finds the unscoped `Order.find`
in about the time it takes to read the line, and cross-checks against the spec: AC5 explicitly
requires a 404 for a foreign order, and no RSpec example proves it either way.

**Reviewer verdict:**
> **CHANGES REQUESTED.**
> 1. `app/controllers/checkout/orders_controller.rb:26` — `Order.find(params[:id])` is not scoped to
>    `Current.user`. Any signed-in user can read any other user's order by incrementing the id in the
>    URL (IDOR / broken access control, OWASP A01:2021). This is exactly the failure mode
>    `docs/features/FEATURE-2-checkout.md` AC5 names by name — "not satisfied by 'the route requires
>    sign-in' alone." Fix: `Current.user.orders.find(params[:id])`.
> 2. `spec/requests/checkout_spec.rb` — AC5's negative case (a second user attempting to read the
>    first user's order) has no RSpec example. Green RSpec here is not evidence of anything: add the
>    case, confirm it fails against the current code, then confirm the fix in finding 1 makes it
>    pass.
> Both findings block merge. RuboCop and Brakeman being clean is not sufficient — see this project's
> `docs/research/NOTE-SDLC-4-3-brakeman-checks.md`, which says plainly that Brakeman does not cover
> this class of bug; that's what a fresh reviewer is *for*.

### Stage 3, again — the fix

**Architect routes the verdict back to the implementer** (the loop's second exit ramp,
`docs/architecture.md` §3 step 5 → step 3). The implementer:

1. Adds the missing negative RSpec example to `spec/requests/checkout_spec.rb` — exactly the
   `"does NOT return another user's order — the IDOR case"` example committed in
   [`code/rails-estore/spec/requests/checkout_spec.rb`](../code/rails-estore/spec/requests/checkout_spec.rb)
   — and confirms it fails against the still-broken controller:

```text
[REFERENCE — illustrative]
Failures:
  1) Checkout GET /checkout/orders/:id — authorization (AC5) does NOT return another user's order — the IDOR case
     Failure/Error: get checkout_order_path(others_order)
       expected ActiveRecord::RecordNotFound but nothing was raised
1 example, 1 failure
```

2. Applies the one-line fix: `Order.find(params[:id])` → `Current.user.orders.find(params[:id])`
   (the version committed in
   [`app/controllers/checkout/orders_controller.rb`](../code/rails-estore/app/controllers/checkout/orders_controller.rb)).
3. Re-runs the gate: RSpec, RuboCop, and Brakeman all report green, and this time the AC5 negative
   case is actually one of the examples making RSpec green.

### Stage 5, again — Review (APPROVE)

**The same fresh reviewer** re-checks its own two findings against the new diff: the query is now
scoped, the negative example exists and is confirmed to have failed before the fix (the log above).
It re-runs the full gate independently once more.

**Reviewer verdict:**
> **APPROVE.** Both findings from the previous round are resolved: `show` is scoped to
> `Current.user.orders`; the negative RSpec example exists, is confirmed to have failed against the
> old code, and passes against the new code. AC1–AC6 all map to passing examples. RSpec/RuboCop/
> Brakeman are clean. No file outside `FEATURE-2`'s "Assets to produce" changed.

### Stage 6 — Merge

**Architect merges.** PR body maps every acceptance criterion to its evidence, and — unlike
Part A's straight-line pass — explicitly notes the one review round that caught the IDOR, because
`docs/definition-of-done.md`'s process section asks for exactly that: the evidence trail, not just
the final green checkmark.

---

---

## Part C — FEATURE-3: product catalog (two specialist defects, caught and fixed)

> Added by SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents. Same convention as Parts A and B: real file
> names, real spec, real committed code, real review reasoning; illustrative console bytes for any
> RSpec/html-proofer/axe run, because this sandbox has neither a Ruby toolchain nor a browser (no
> `chromedriver` on `PATH` here either — see [`03-rails-estore-sdlc.md`](../03-rails-estore-sdlc.md)'s
> Environment note).

### Stage 1 — Request → spec

**Owner prompt to the architect:**
> "Sign-up and checkout work. Now make the actual product pages something a search engine can index
> and something a screen-reader user can use — right now there isn't even a layout file."

**Architect** writes
[`docs/features/FEATURE-3-product-catalog-seo-a11y.md`](../code/rails-estore/docs/features/FEATURE-3-product-catalog-seo-a11y.md):
nine acceptance criteria (AC1–AC9) covering semantic/labelled markup, unique titles, Open Graph,
Product + BreadcrumbList JSON-LD, `robots.txt`/sitemap, and the axe/html-proofer gate — and, in
`docs/architecture.md` terms, names two NEW reviewer roles this feature exists to exercise:
`seo-optimizer.md` and `frontend-qa.md`, dispatched alongside the general `reviewer.md`, not instead
of it.

### Stage 2 — Ground the unknowns

All four claims the spec cites were already grounded: `NOTE-SDLC-4-ADD-1-gem-npm-versions.md` (the
five gems + `@lhci/cli`, pinned), `NOTE-SDLC-4-ADD-2-schema-product.md` (the Product JSON-LD shape),
`NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md` (the WCAG checks axe enforces), `NOTE-SDLC-4-ADD-4-robots-sitemap-og.md`
(`robots.txt`/sitemap/Open Graph). No new dispatch needed.

### Stage 3 — Failing test first, then implement (two slips, one per specialist's lane)

**Implementer, step 1:** writes `spec/system/accessibility_spec.rb` and `spec/system/seo_spec.rb`
against stub views, confirms both fail for the right reason (no layout, no meta tags, no JSON-LD
exist yet).

**Implementer, step 2:** writes the real
[`ProductsController`](../code/rails-estore/app/controllers/products_controller.rb),
[`ProductsHelper`](../code/rails-estore/app/helpers/products_helper.rb), and the layout. Two slips
ship, one in each new specialist's lane:

```erb
<%# app/views/products/index.html.erb — AS FIRST WRITTEN (the planted frontend-qa slip) %>
<%= form_with url: products_path, method: :get, role: "search" do |f| %>
  <div>
    <%= f.search_field :q, value: params[:q], placeholder: "Search products" %>
    <%= f.submit "Search" %>
  </div>
<% end %>
```

No `<%= f.label :q, ... %>` — a placeholder is not a label (WCAG 1.3.1 / 4.1.2;
`NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md`'s own quote: "Placeholder text is not enough — it disappears
as soon as you type and is not announced reliably"). And:

```erb
<%# app/views/products/show.html.erb — AS FIRST WRITTEN (the planted seo-optimizer slip) %>
<h1><%= @product.name %></h1>
<p><%= @product.description %></p>
<p>$<%= "%.2f" % @product.price_dollars %></p>
<%# ... no <script type="application/ld+json"> block at all %>
```

No Product JSON-LD anywhere on the page.

### Stage 4 — Gate (green on what it could run)

```text
[REFERENCE — illustrative]
$ bundle exec rspec spec/models spec/requests
..........................
26 examples, 0 failures

$ bundle exec rspec spec/system/accessibility_spec.rb spec/system/seo_spec.rb
[verify] no chromedriver on PATH in this environment — system specs requiring `js: true` cannot run
here; reproduce on a machine with Chrome + chromedriver installed (README.md, "Frontend gate").
2 examples, 0 failures, 2 not run
```

`bundle exec rubocop` and `bundle exec brakeman -q --no-summary` both report clean — style has no
opinion on a missing `<label>`, and Brakeman's checks don't cover either WCAG or SEO categories at
all [source: `NOTE-SDLC-4-3-brakeman-checks.md`, same "not in its catalog" gap this chapter's IDOR
case already showed for authorization]. The implementer reports every gate it could actually run as
green and hands off for review, honestly believing FEATURE-3 is done.

### Stage 5 — Review (two CHANGES REQUESTED, in parallel)

**Architect dispatches three reviewers**: the general `reviewer.md` (fidelity/test-first/grounded/
scope — nothing to flag here, there's no `Current.user`-scoped query or `permit()` call on this
controller at all), and — new to this feature — `seo-optimizer.md` and `frontend-qa.md` in parallel.
Both actually run their tools rather than trusting the "system specs didn't run" note above.

**`frontend-qa` verdict:**
> **CHANGES REQUESTED.**
> 1. `app/views/products/index.html.erb` — the search `<input type="search" name="q">` has no
>    associated `<label>`, `aria-label`, or `aria-labelledby`. axe rule `label`, impact **critical**,
>    WCAG 1.3.1 / 4.1.2. A screen-reader user hears "edit text, blank" with no indication of what to
>    type. Fix: add `<%= f.label :q, "Search products" %>` before the field.
>
> Coverage note: axe automated checks catch ~30–40% of real WCAG issues
> (`NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md`); I also read the rendered markup by hand and found nothing
> else — that is not the same claim as "this page is fully accessible."

**`seo-optimizer` verdict:**
> **CHANGES REQUESTED.**
> 1. `app/views/products/show.html.erb` — no `<script type="application/ld+json">` Product block
>    anywhere on the page. Google Merchant Listing requires `name`, `image`, `offers.price`,
>    `offers.priceCurrency` at minimum (`NOTE-SDLC-4-ADD-2-schema-product.md`); without it this page
>    is not eligible for a merchant listing rich result at all, and AC5 is unmet. Fix: render
>    `product_json_ld(@product)` (already defined in `ProductsHelper`, just not called from the view).

### Stage 3, again — the fix

**Implementer** adds the missing `<%= f.label :q, "Search products" %>` to `index.html.erb`, and adds
the two `<script type="application/ld+json">` lines (Product + BreadcrumbList) to `show.html.erb` —
exactly the versions committed under
[`app/views/products/`](../code/rails-estore/app/views/products/). Re-runs the gate:

```text
[REFERENCE — illustrative, on a machine with chromedriver on PATH]
$ bundle exec rspec spec/system/accessibility_spec.rb spec/system/seo_spec.rb
....
4 examples, 0 failures

$ bundle exec rake html_proofer:check
Running ["ScriptCheck", "LinkCheck", "ImageCheck"] on ["tmp/html_proofer/index.html", "tmp/html_proofer/show.html"] ... 

HTML-Proofer finished successfully.
```

### Stage 5, again — Review (APPROVE, both specialists + the general reviewer)

`frontend-qa` re-runs `accessibility_spec.rb` and confirms the label fix resolves the axe finding.
`seo-optimizer` re-parses the rendered JSON-LD and confirms `name`/`image`/`offers.price`/
`offers.priceCurrency` are all present and valid, plus the recommended `sku`/`brand`/`availability`.
The general `reviewer.md` re-confirms fidelity and scope. All three: **APPROVE**.

### Stage 6 — Merge

**Architect merges.** PR body maps AC1–AC9 to their evidence, and names both specialist sign-offs
alongside the general reviewer's — `docs/definition-of-done.md`'s SEO & Accessibility section
requires both, not either.

---

## What these three transcripts are actually showing

Part A is what "the loop worked" looks like: spec → ground → test-first → green gate → clean
review → merge, no surprises. Part B is the security lesson: an agent under no malicious intent,
following the spec, writing real tests, and still shipping a real vulnerability, because three
automated gates all had a legitimate reason to stay green — the fresh reviewer's authorization-by-hand
step, a deliberate instruction in
[`.claude/agents/reviewer.md`](../code/rails-estore/.claude/agents/reviewer.md), is what caught it.
Part C makes the same point about a *different* pair of quality dimensions: RuboCop and Brakeman have
no SEO or accessibility checks in their catalog at all — not a gap in an otherwise-broad scanner, a
category they were never built to cover — so a missing `<label>` and a missing Product JSON-LD block
sail through every gate this project had *before* this addendum. `seo-optimizer.md` and
`frontend-qa.md` are what closes that gap, the same way a fresh reviewer closed the authorization gap
in Part B: a named specialist, with a checklist, instructed to check by hand what no available tool
checks automatically. That is the chapter's whole argument, now proven on three different classes of
defect: governance is what makes AI-assisted development safe, not the model's good intentions, and
not even a good automated gate alone.
