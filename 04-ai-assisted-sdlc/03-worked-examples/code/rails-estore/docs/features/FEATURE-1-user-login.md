# FEATURE-1: User sign-up, login, and logout

**Status:** approved
**Owner:** architect (Opus)
**Routing:** implementer=Sonnet · research=NOTE-SDLC-4-1, NOTE-SDLC-4-2, NOTE-SDLC-4-5 · review=Sonnet (fresh)

## Intent

Before there can be a cart or a checkout, there has to be an account. Give a visitor a way to sign
up, sign in, and sign out, using Rails 8's native authentication generator shape
(`User` + `has_secure_password`, a database-tracked `Session`, `Current`) rather than an external
gem — the dependency-light, officially-maintained option for a new Rails 8 app
[source: NOTE-SDLC-4-2-auth-generator.md]. The generator itself does not ship a sign-up flow
[source: NOTE-SDLC-4-2-auth-generator.md, "Caveats"] — this feature adds `RegistrationsController`
on top of it.

## Acceptance criteria

- AC1 — `POST /registration` with a valid `email_address`, `password`, and `password_confirmation`
  creates a `User`, starts a session (a `Session` row + a signed permanent cookie), and redirects to
  the root path.
- AC2 — the created `User`'s `password_digest` column never equals, and cannot be reversed to, the
  plaintext password submitted — `has_secure_password` bcrypt-hashes it
  [source: NOTE-SDLC-4-5-auth-security.md].
- AC3 — `POST /registration` with an extra, non-permitted parameter (`admin: "true"`) creates the
  user WITHOUT setting `admin` — the registration controller's `permit()` list does not include it
  [source: NOTE-SDLC-4-5-auth-security.md, mass-assignment].
- AC4 — `POST /session` with correct credentials for an existing user signs them in (a `Session` row
  created, cookie set, redirect to the page they were trying to reach or root); with incorrect
  credentials it re-renders the sign-in form with an error and creates NO session.
- AC5 — `DELETE /session` destroys the current `Session` row and clears the cookie.
- AC6 — a signed-out request to a page that requires authentication (`GET /cart`) redirects to
  `new_session_path` rather than rendering the page — the `Authentication` concern's
  `require_authentication` before-action, applied by default in `ApplicationController`
  [source: NOTE-SDLC-4-2-auth-generator.md].

## Claims to ground

- Rails 8's native auth generator shape (User/Session/Current, has_secure_password, signed
  cookie) — grounded, `docs/research/NOTE-SDLC-4-2-auth-generator.md`.
- bcrypt version and CVE status, `has_secure_password`'s hashing behaviour — grounded,
  `docs/research/NOTE-SDLC-4-1-versions.md` and `docs/research/NOTE-SDLC-4-5-auth-security.md`.
- Strong parameters as the mass-assignment defence — grounded,
  `docs/research/NOTE-SDLC-4-5-auth-security.md`.

## Out of scope

- Password reset (the generator's `PasswordsMailer` exists but wiring an actual mailer/SMTP is a
  separate feature).
- Email verification / confirmable — not part of the native generator; Devise territory if ever
  needed (`docs/architecture.md` decision log).
- Rate-limiting login attempts beyond Rails' documented `rate_limit` action helper — a hardening
  pass, not this feature.

## Assets to produce

- `app/models/user.rb`, `app/models/session.rb`, `app/models/current.rb`
- `app/controllers/concerns/authentication.rb`
- `app/controllers/application_controller.rb`
- `app/controllers/sessions_controller.rb`, `app/controllers/registrations_controller.rb`
- `app/views/sessions/new.html.erb`, `app/views/registrations/new.html.erb`
- `db/schema.rb` (users, sessions tables)
- `spec/models/user_spec.rb`, `spec/requests/registrations_spec.rb`, `spec/requests/sessions_spec.rb`

## Gates

Entry: this spec approved; the three grounding notes above landed. Exit:
`docs/definition-of-done.md` checklist, in full — including the mass-assignment and password-hashing
security cases (AC2, AC3) with an explicit RSpec example each.
