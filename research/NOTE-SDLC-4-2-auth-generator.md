# NOTE-SDLC-4-2: Rails 8 built-in authentication generator—shape and Devise comparison

**Answer:**

Rails 8's built-in `bin/rails generate authentication` exists and is the current idiomatic authentication scaffold for new Rails apps. It generates:
- **User model** with `has_secure_password` (bcrypt-based) and email_address field (unique indexed)
- **Session model** with signed token (database-tracked), ip_address, and user_agent fields; uses `has_secure_token` for token generation
- **Current** class (inherits `ActiveSupport::CurrentAttributes`) providing per-request access to the current user
- **Authentication concern** with `require_authentication` filter and signed session cookies
- **PasswordsMailer** for password reset tokens (default 15-minute expiry)
- Controllers for sign-up, login, logout, and password reset

**vs. Devise:** Devise is a heavier, external gem with Warden-based architecture, mountable engine, and 12+ optional modules (confirmable, lockable, omniauthable, etc.). Rails' native generator is dependency-light, simpler, and now receives official documentation and framework maintenance. **Recommendation: Use Rails' native generator for teaching** — it demonstrates idiomatic Rails patterns, reduces external dependencies, and suffices for the login/logout/password-reset scope of this chapter.

**Evidence:**

1. **Generator command exists** — [Rails 8 introduces a basic authentication generator](https://www.bigbinary.com/blog/rails-8-introduces-a-basic-authentication-generator) confirms `bin/rails generate authentication` available in Rails 8+.

2. **Generator shape (User, Session, Current)** — [Rails 8 adds built in authentication](https://blog.saeloun.com/2025/05/12/rails-8-adds-built-in-authentication-generator/) details: "The command generates a User and Session models and a Current class... Sessions table includes unique token, ip_address and user_agent fields... Current is a singleton inheriting ActiveSupport::CurrentAttributes."

3. **Session model and signed cookies** — [Rails 8 Authentication with the auth generator](https://avohq.io/blog/rails-8-authentication) states: "Instead of depending solely on session cookies, Rails 8's auth system persists session records in the database using a token stored in a signed cookie."

4. **has_secure_password, password reset** — [Built-In Authentication in Rails 8: Deep Dive and Comparison](https://andriifurmanets.com/blogs/built-in-authentication-in-rails): "User model holds user credentials and leverages has_secure_password for bcrypt hashing... Two key methods have been added: password_reset_token (generates default configuration for 15-minute password reset token) and find_by_password_reset_token (finds user by given token or returns nil)."

5. **Idiomatic for new Rails 8 apps** — [Rails 8 release notes](https://guides.rubyonrails.org/8_0_release_notes.html) state: "A new generator creates a session-based authentication system with password reset capabilities." This is now the official Rails scaffold; Devise is no longer the default starting point for new projects.

6. **Devise comparison** — [Rails 8 Authentication: Why the New Built-in Generator Matters (and What It Means for Devise)](https://rubystacknews.com/2026/02/16/rails-8-authentication-why-the-new-built-in-generator-matters-and-what-it-means-for-devise/): "Devise is the most widely used authentication library in the Rails ecosystem [and] integrates deeply with Rails [with] Warden-based [architecture]... Since the native auth is now part of Rails core, it will receive official documentation and maintenance, making it an attractive option for new apps." Also notes: "Rails 8's auth system tracks full session history, whereas Devise's trackable module only stores the most recent session info."

7. **Official Rails guides** — [Rails 8.0 Release Notes](https://guides.rubyonrails.org/8_0_release_notes.html) documentation confirms the authentication generator as part of Rails core.

**Caveats / limits:**

- The generator provides the foundation; the chapter writer must add sign-up (user registration) logic — the generator does not include a sign-up route or form by default, only sign-in, logout, and password reset.
- If the chapter needs advanced features (e.g., 2FA, social login, confirmable, lockable), Devise or a complementary gem is recommended; the native generator intentionally keeps those out-of-scope for simplicity.
- The Current class is a **singleton per request** and not directly tied to database state; it is populated by the authentication concern filter on each request.
- Date checked: **2026-09-04**.

**Recommendation:**

Teach the native `bin/rails generate authentication` generator. It is:
1. **Official** — part of Rails core as of 8.0 (November 2024), with continued maintenance.
2. **Simpler for learning** — zero external dependencies (Devise adds Warden, Bcrypt, etc.); the chapter focuses on governance gates, not gem mastery.
3. **Sufficient** — covers sign-up, login, logout, password reset, and session tracking for the e-commerce app scope.
4. **Honest** — shows readers what modern Rails looks like out of the box, vs. a third-party convention.

If a reader later adds features beyond scope (e.g., omniauthable), Devise or similar is an explicit enhancement, not a hidden dependency from day 1.
