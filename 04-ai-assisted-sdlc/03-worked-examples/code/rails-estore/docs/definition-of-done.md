# Definition of Done — feature gate checklist

A feature is DONE only when **every** box below is checked. No exceptions "to be fixed later" — on
an app that stores passwords and talks to a payment gateway, "later" is how a mass-assignment hole
or a missing authorization check reaches production.

## Fidelity to the spec
- [ ] Every acceptance criterion in the approved `docs/features/FEATURE-*.md` is met.
- [ ] Anything cut from the spec is recorded in the spec's "Out of scope", not silently dropped.

## Test-first (the implementer's discipline)
- [ ] An RSpec example encoding the acceptance criteria was written and run BEFORE the production
      code that satisfies it — and it failed for the right reason (missing behaviour, not a typo).
- [ ] No test was weakened (loosened assertion, deleted case, `skip`/`pending`) to make the gate pass.

## Grounded
- [ ] Every gem version pinned in `Gemfile`, every library-behaviour claim the design relies on
      (the auth generator's shape, Brakeman's check categories, Stripe's Ruby API), and every
      CVE/security check traces to a researcher's note or a live, dated citation.
- [ ] No claim rests on model memory alone.

## Security (the checks this app's threat model actually needs)
- [ ] **Authorization:** every controller action that reads or writes a record scoped to a user
      (`Order`, `Cart`, `Session`) looks it up through `Current.user`, never through a bare
      `Model.find(params[:id])`. Verified by an explicit RSpec case: a second user cannot read or
      modify the first user's record via a guessed/known id.
- [ ] **Mass assignment:** every `permit()` call whitelists only the attributes the feature needs.
      Verified by an explicit RSpec case: submitting an extra privileged attribute (e.g.
      `admin: true`) through a public form does NOT set it.
- [ ] **Password hashing:** verified by an explicit RSpec case that the stored `password_digest`
      never equals (and cannot be reversed to) the plaintext password.
- [ ] **No secrets:** nothing resembling a live key (`sk_live_`/`pk_live_`/`rk_live_`) anywhere in
      the diff; `.env.example` documents every variable with a placeholder only.
- [ ] `bundle exec brakeman -q --no-summary` reports zero High-confidence warnings (Medium/Low
      warnings are triaged, not silently ignored — documented in a comment or `.brakeman.yml` if
      suppressed).

## Green gate
- [ ] `bundle exec rspec` passes, all examples green.
- [ ] `bundle exec rubocop` passes clean.
- [ ] `bundle exec brakeman -q --no-summary` passes the High-confidence threshold above.
- [ ] `.claude/hooks/verify.sh` (the fast per-edit subset) is green.

## Hygiene
- [ ] No secrets committed; `.env.example` updated if config changed.
- [ ] Nothing outside the feature spec's stated files changed.

## Process
- [ ] One feature per PR; PR body maps each acceptance criterion → its evidence (RSpec example name
      / gate log / grounding note).
- [ ] Independent review by a **fresh** reviewer (not the implementer) — sign-off recorded, with
      authorization and mass-assignment explicitly checked, not assumed from the spec's intent.
- [ ] Architect (Opus) merge approval.

## Escalate instead of forcing
Stop and ask the owner if a feature's scope is ambiguous, a claim can't be grounded from available
sources, a gem turns out to have a known CVE or is unmaintained, or a design choice conflicts with
existing code — especially anything touching authentication, authorization, or payment.
