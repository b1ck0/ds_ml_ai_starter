# Validation log — rails-estore/ scaffold

Real, captured output from validating every governance file under
[`code/rails-estore/`](../code/rails-estore/), plus a real secret-scan across the whole tree, run in
this repository's own sandbox (Windows 11, Python 3.13, Git Bash) while writing this chapter — the
same sandbox and the same method the Java chapter's
[`validation-log.md`](../../02-local-environment-setup/../03-worked-examples/artefacts/validation-log.md)
used. **§§1–3 and §5 below are real, captured evidence** — every command was actually executed.
**§4 and §6 are illustrative** — this repository has no Ruby/Rails toolchain (`bundle`, `rspec`,
`rubocop`, `brakeman` are not installed here) and no browser/chromedriver/Node either, so
RSpec/RuboCop/Brakeman (§4) and axe/html-proofer/Lighthouse CI (§6, added by
SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents) output is presented as a grounded reference (exact
command, exact pinned versions from `research/NOTE-SDLC-4-1-versions.md` and
`research/NOTE-SDLC-4-ADD-1-gem-npm-versions.md`), not a captured run — see
[`03-rails-estore-sdlc.md`](../03-rails-estore-sdlc.md)'s Environment note for why.

## 1. `settings.json` parses as valid JSON

```
$ .venv/Scripts/python.exe -m json.tool code/rails-estore/.claude/settings.json
{
    "$schema": "https://json.schemastore.org/claude-code-settings.json",
    "hooks": {
        "PostToolUse": [
            {
                "matcher": "Edit|Write",
                "hooks": [
                    { "type": "command", "command": ".claude/hooks/verify.sh" }
                ]
            }
        ],
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    { "type": "command", "command": ".claude/hooks/guard.sh" }
                ]
            }
        ],
        "SessionStart": [
            {
                "hooks": [
                    { "type": "command", "command": ".claude/hooks/context.sh" }
                ]
            }
        ]
    }
}
```

Same three-event, `{matcher, hooks:[{type:"command", command:...}]}` shape as
`ds_ml_ai_starter`'s own `.claude/settings.json` and `java-project`'s port — the schema
(`json.schemastore.org/claude-code-settings.json`) does not know or care what language the hooks
check. `python -m json.tool` exits non-zero on malformed JSON; a clean pretty-printed echo, as above,
is the pass signal.

## 2. Agent frontmatter is well-formed YAML and matches the documented schema

`code/validate_frontmatter.py` (parses the `---`-delimited frontmatter with PyYAML, checks `name` +
`description` are present, `model` is one of the documented values, every other key is in the
documented optional-field set) run against all five roster files:

```
$ .venv/Scripts/python.exe code/validate_frontmatter.py \
    code/rails-estore/.claude/agents/researcher.md \
    code/rails-estore/.claude/agents/implementer.md \
    code/rails-estore/.claude/agents/reviewer.md \
    code/rails-estore/.claude/agents/seo-optimizer.md \
    code/rails-estore/.claude/agents/frontend-qa.md
OK   code/rails-estore/.claude/agents/researcher.md: name='researcher' model='haiku' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/implementer.md: name='implementer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/reviewer.md: name='reviewer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/seo-optimizer.md: name='seo-optimizer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/frontend-qa.md: name='frontend-qa' model='sonnet' fields=['description', 'model', 'name', 'tools']
```

Model routing matches `docs/architecture.md` §2 (researcher on Haiku, implementer/reviewer on
Sonnet) plus the two specialists this addendum adds (`seo-optimizer`, `frontend-qa`, both Sonnet) —
see `CLAUDE.md`'s Model routing section.

## 3. Hooks actually execute and enforce what they claim to

`context.sh` (`SessionStart`) run directly:

```
$ bash code/rails-estore/.claude/hooks/context.sh
──────────────────────────────────────────────────────────
 rails-estore — a governed SDLC scaffold for a Rails e-store
──────────────────────────────────────────────────────────
 Read first: docs/architecture.md · docs/definition-of-done.md · CLAUDE.md
 Roles: Opus (architect, main session) · .claude/agents/researcher.md (Haiku) ·
        .claude/agents/implementer.md (Sonnet) · .claude/agents/reviewer.md (fresh Sonnet)

 Feature specs:
   • FEATURE-1: User sign-up, login, and logout                  approved
   • FEATURE-2: Cart to order — checkout with a stubbed payment seam  approved

 Gate reminders: failing RSpec example committed before the code it covers ·
 bundle exec rspec passes · rubocop clean · brakeman -q --no-summary clean ·
 no live Stripe key (sk_live_/pk_live_) anywhere · independent review before merge.
 (docs/definition-of-done.md)
──────────────────────────────────────────────────────────
exit=0
```

`guard.sh` (`PreToolUse`), fed synthetic hook payloads on stdin exactly as Claude Code would — five
cases, real output:

```
$ printf '{"command":"bundle exec rspec"}' | bash guard.sh
exit=0

$ printf '{"command":"rm -rf /"}' | bash guard.sh
[guard] BLOCKED: rm -rf /
exit=2

$ printf '{"command":"git commit --no-verify -m test"}' | bash guard.sh
[guard] BLOCKED: git commit --no-verify (skips the RSpec/RuboCop/Brakeman pre-commit gate)
exit=2

$ printf '{"command":"curl -H XSomeKey:sk_live_EXAMPLE_redacted_not_a_real_key https://example.com"}' | bash guard.sh
[guard] BLOCKED: a live Stripe key pattern (sk_live_/pk_live_/rk_live_) appeared in a command — use pk_test_/sk_test_ only, from .env, never inline
exit=2

$ printf '{"command":"echo STRIPE_SECRET_KEY=$STRIPE_SECRET_KEY"}' | bash guard.sh
[guard] BLOCKED: printing a secret to stdout
exit=2
```

A benign build command passes through (exit 0); a destructive filesystem command, a hook-skipping
commit, a **live** Stripe key pattern, and printing a secret env var are all vetoed (exit 2). Note
what does NOT trip the live-key rule: every `pk_test_`/`sk_test_` placeholder used throughout this
project's code and docs is untouched by it — the regex matches only the `_live_` infix.

`verify.sh` (`PostToolUse`), same synthetic-payload approach:

```
$ printf '{"file_path":"app/controllers/checkout/orders_controller.rb"}' | bash verify.sh
[verify] bundle not on PATH — skipping gate for app/controllers/checkout/orders_controller.rb
exit=0

$ printf '{"file_path":"README.md"}' | bash verify.sh
[verify] no checks for: README.md
exit=0

$ printf '{"file_path":"Gemfile"}' | bash verify.sh
[verify] bundle not on PATH — skipping check
exit=0
```

`verify.sh` degrades gracefully when `bundle` is absent (this sandbox) rather than failing the whole
hook — the same "skip gracefully when a tool is absent" contract `java-project/.claude/hooks/verify.sh`
documents for a missing `mvn`.

## 4. RSpec / RuboCop / Brakeman — reference reproduction (not run in this sandbox)

No Ruby/Rails toolchain is installed here (`which bundle`, `which rspec` both resolve to nothing).
The commands, file names, and expected outcomes below are grounded in
`research/NOTE-SDLC-4-1-versions.md` (pinned gem versions) and
`research/NOTE-SDLC-4-3-brakeman-checks.md` (Brakeman's real check categories and output shape);
reproduce for real on a machine with Ruby 4.0.6 + Rails 8.1.3.1 installed, from inside
`code/rails-estore/` after `bundle install`.

```text
[REFERENCE — illustrative RSpec output, gem versions per NOTE-SDLC-4-1-versions.md]
$ bundle exec rspec
.........................

Finished in 1.84 seconds (files took 2.11 seconds to load)
25 examples, 0 failures

Randomized with seed 48213
```

```text
[REFERENCE — illustrative RuboCop output]
$ bundle exec rubocop
Inspecting 24 files
........................

24 files inspected, no offenses detected
```

```text
[REFERENCE — illustrative Brakeman output shape, per NOTE-SDLC-4-3-brakeman-checks.md's documented
 output format — text/JSON/11 formats total]
$ bundle exec brakeman -q --no-summary

== Brakeman Report ==

No warnings found
```

### A Brakeman warning, caught and resolved (the case Brakeman IS built for)

Early in FEATURE-2's development, `app/views/products/show.html.erb` had one line different from the
version committed under `code/rails-estore/`:

```erb
<%# EARLY DRAFT — flagged and then fixed, not what's committed %>
<p><%= @product.description.html_safe %></p>
```

`.html_safe` tells Rails "do not HTML-escape this string" — correct for content the app itself fully
controls, wrong for anything that traces back to user or admin input, because it turns a stored
string directly into raw HTML the browser executes. `docs/research/NOTE-SDLC-4-3-brakeman-checks.md`
names this exact pattern as one of Brakeman's core checks:

```text
[REFERENCE — illustrative, matching Brakeman's documented "Cross Site Scripting" warning shape]
$ bundle exec brakeman -q --no-summary

== Brakeman Report ==

+ Confidence: High
  Category: Cross Site Scripting
  Check: CrossSiteScripting
  Message: Unescaped model attribute
  File: app/views/products/show.html.erb
  Line: 3
  Code: @product.description.html_safe

1 warning found
```

`.claude/hooks/verify.sh`'s `brakeman -q --no-summary` step fails non-zero on this — the fix (drop
`.html_safe`; ERB HTML-escapes by default, which is what the committed
[`app/views/products/show.html.erb`](../code/rails-estore/app/views/products/show.html.erb) does)
brings Brakeman back to zero warnings before the implementer reports the gate green. Contrast this
with the checkout loop transcript's IDOR finding, which Brakeman does **not** catch — the two
together are the honest picture: Brakeman is real and useful for the vulnerability classes it
targets (SQLi, mass assignment, XSS), and a fresh human-directed reviewer is still required for the
classes it doesn't (broken access control across records of the same type).

## 5. Secret scan — real, captured, clean

Required before this chapter could report done: grep the whole `rails-estore/` tree for anything
resembling a real key.

```
$ grep -rniE 'sk_live|pk_live|rk_live' code/rails-estore/
code/rails-estore/.env.example:2:# ... NEVER put a live key (sk_live_/pk_live_/rk_live_) here ...
code/rails-estore/.claude/agents/reviewer.md:39:7. **Secrets:** grep the diff for anything resembling a real key (`sk_live`, `pk_live`, ...
code/rails-estore/.claude/hooks/guard.sh:27:  deny "a live Stripe key pattern (sk_live_/pk_live_/rk_live_) appeared ..."
code/rails-estore/.claude/hooks/context.sh:28:echo " no live Stripe key (sk_live_/pk_live_) anywhere ..."
code/rails-estore/docs/definition-of-done.md:32:- [ ] **No secrets:** nothing resembling a live key (`sk_live_`/`pk_live_`/`rk_live_`) ...
code/rails-estore/docs/architecture.md:75:- **No secrets:** nothing resembling a live key (`sk_live_`/`pk_live_`/`rk_live_`) ...

$ grep -rnE '[sp]k_(live|test)_[A-Za-z0-9]{10,}' code/rails-estore/
(no matches)
```

Every `sk_live`/`pk_live`/`rk_live` hit from the first grep is prose *describing* the rule (in a hook,
a doc, or the reviewer's own checklist) — none is an actual key value. The second grep, which looks for
anything shaped like a real key (a `pk_/sk_` prefix followed by a long run of key characters), returns
**nothing**: `.env.example` uses spelled-out placeholders (`sk_test_your_secret_key_here`), not a
key-shaped string, so there is nothing for a scanner to trip on. **Clean.**

> **Real-world footnote.** An earlier draft of this very file used `sk_test_` followed by 24 literal
> `X`s as the placeholder. GitHub's own **push protection** rejected the push anyway — its Stripe
> detector matches on *shape*, and `sk_test_` + 24 characters is that shape, X's or not. The fix was
> to use a spelled-out placeholder with no key-length character run. That is the same defence-in-depth
> this chapter preaches, caught one layer further out than our own `guard.sh` — exactly the point.

## 6. Frontend gate — axe / html-proofer / Lighthouse CI reference output (added by this addendum)

Same convention as §4: no browser (no `chromedriver`) and no Node/`npx` are installed in this
sandbox, so every block below is a **grounded reference reproduction**, not a captured run — exact
commands and exact pinned versions from `docs/research/NOTE-SDLC-4-ADD-1-gem-npm-versions.md`.
Reproduce for real on a machine with Ruby 4.0.6 + Chrome/chromedriver + Node 24 installed, from
inside `code/rails-estore/` after `bundle install` and `npm install` (see `README.md`, "Frontend
gate").

**axe, catching the FEATURE-3 planted a11y slip** (before the fix in
[`app/views/products/index.html.erb`](../code/rails-estore/app/views/products/index.html.erb) — see
the loop transcript, Part C):

```text
[REFERENCE — illustrative axe violation shape, per NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md's documented
 axe output fields (id, impact, description, help, helpUrl, nodes, tags)]
$ bundle exec rspec spec/system/accessibility_spec.rb
F.

Failures:

  1) Product catalog accessibility has no automatically detectable WCAG 2.1 AA violations on the catalog index
     Failure/Error: expect(page).to be_axe_clean.according_to(:wcag21aa)
       expected no accessibility violations, but got 1:

       Violation: label (critical)
         Form elements must have labels
         https://dequeuniversity.com/rules/axe/4.13/label
         Node: input[name="q"]
         Tags: wcag2a, wcag412, wcag131, section508

2 examples, 1 failure
```

**axe, after the fix:**

```text
[REFERENCE — illustrative]
$ bundle exec rspec spec/system/accessibility_spec.rb
..

2 examples, 0 failures
```

**html-proofer**, run via `lib/tasks/html_proofer.rake` (5.2.2 — programmatic
`HTMLProofer.check_directory(...).run`, exactly `NOTE-SDLC-4-ADD-1-gem-npm-versions.md`'s cited
usage):

```text
[REFERENCE — illustrative]
$ bundle exec rake html_proofer:check
Running ["ScriptCheck", "LinkCheck", "ImageCheck"] on ["tmp/html_proofer/index.html", "tmp/html_proofer/show.html"] ...

HTML-Proofer finished successfully.
```

**Lighthouse CI** (`@lhci/cli` 0.15.1, npm — Node reference gate only; not invoked by
`.claude/hooks/verify.sh`, which is a Ruby-only script):

```text
[REFERENCE — illustrative Lighthouse CI summary shape]
$ npm install
$ npx lhci autorun
✅  1 result(s) for http://localhost:3000/products
Accessibility:    98/100
SEO:              100/100
Best Practices:   96/100
Performance:      89/100

No LHCI budget assertions failed.
```

Contrast this section with §4's Brakeman example the same way the chapter's Part B/Part C pairing
does: automated tools are real and each catches a real class of defect (a `.html_safe` XSS for
Brakeman; a missing `<label>` for axe) — and each has a boundary drawn by what it checks at all,
which is exactly why `seo-optimizer.md` and `frontend-qa.md` exist as named, by-hand review steps
rather than "wait for the tool to catch it."

## Summary

| Check | Result |
|---|---|
| `settings.json` well-formed JSON | PASS (real) |
| 5× agent frontmatter well-formed + schema-valid | PASS (5/5, real) |
| `context.sh` runs clean | PASS (exit 0, real) |
| `guard.sh` allows benign, blocks `rm -rf /`, `--no-verify`, a live Stripe key, and printing a secret | PASS (5/5 cases, real) |
| `verify.sh` degrades gracefully without `bundle` | PASS (3/3 cases, real) |
| Secret scan across `code/rails-estore/` | PASS — clean (real) |
| `bundle exec rspec` / `rubocop` / `brakeman -q --no-summary` | **NOT run — no Ruby/Rails toolchain in this sandbox.** Presented as grounded reference (exact commands, exact pinned versions); reproduce on a machine with Ruby 4.0.6 / Rails 8.1.3.1 installed. |
| axe / html-proofer / `npx lhci autorun` (frontend gate) | **NOT run — no browser/chromedriver/Node in this sandbox.** Presented as grounded reference (§6); reproduce per `README.md`'s "Frontend gate" section. |

Date verified: 2026-09-04.
