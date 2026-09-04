# NOTE-SDLC-4-3: Brakeman static security scanner—standard Rails tool and check categories

**Answer:**

**Brakeman is the standard Rails static security scanner** for detecting vulnerabilities without running code. It ships 86 checks and covers all major OWASP categories relevant to Rails. Three key checks the chapter should reference:

1. **SQL Injection** — detects unescaped user input in SQL queries (e.g., `User.where("email = '#{email}'")`)
2. **Mass Assignment** — detects direct model instantiation without strong parameters (e.g., `User.create(params[:user])` without permit())
3. **Cross-Site Scripting (XSS)** — detects unsafe output in views (e.g., `<%= user.bio %>` without escaping; unsafe `.html_safe` usage)

Additional checks covered: command injection, CSRF, path traversal, unsafe deserialization, remote code execution, unsafe redirects, and more. Brakeman outputs in 11 formats (text, HTML, JSON, SARIF, JUnit, Markdown, CSV, etc.) for integration into CI/CD pipelines.

**Evidence:**

1. **Brakeman is the standard Rails static analyzer** — [Brakeman Security Scanner](https://brakemanscanner.org/) official site states it is "a static analysis security vulnerability scanner for Ruby on Rails applications" and is widely recommended in Rails security guides and CI/CD documentation.

2. **86 checks covering SQL injection, XSS, command injection, CSRF, mass assignment, path traversal, unsafe deserialization, RCE** — [Help Net Security: Brakeman](https://www.helpnetsecurity.com/2026/01/26/brakeman-open-source-vulnerability-scanner-ruby-on-rails/) (January 26, 2026): "The scanner ships 86 checks covering SQL injection, XSS, command injection, CSRF, mass assignment, path traversal, unsafe deserialization, and remote code execution."

3. **SQL Injection check** — [Brakeman - SQL Injection](https://brakemanscanner.org/docs/warning_types/sql_injection/) documents the check for SQL injection vulnerabilities from improper parameterization.

4. **Mass Assignment check** — [Brakeman - Mass Assignment](https://brakemanscanner.org/docs/warning_types/mass_assignment/) warns when user input is directly assigned to model attributes without strong parameters protection.

5. **XSS check (multiple variants)** — [Brakeman Security Scanner warning types](https://brakemanscanner.org/docs/warning_types/) lists XSS, content tag XSS, and JSON-based XSS as distinct checks.

6. **Works with Rails 2.3.x through 8.x; requires Ruby ≥ 3.2.0** — Official Brakeman documentation; no compatibility issue with Rails 8.1 or Ruby 4.0.6.

7. **Multiple output formats** — Official Brakeman site documents "11 formats: text, HTML, JSON, SARIF, JUnit, Markdown, CSV, tabs, CodeClimate, GitHub, and Sonar" for CI/CD integration.

8. **Recommended in Rails SDLC** — [Rails 8.0 adds Brakeman](https://www.shakacode.com/blog/rails-8-adds-brakeman-by-default/) notes Rails 8 includes Brakeman recommendations in best practices.

**Caveats / limits:**

- Brakeman is **static analysis only**; it does not run code or execute business logic. It will not catch logic errors (e.g., authorization gaps that require dynamic analysis or test coverage).
- Brakeman produces **confidence ratings** (High, Medium, Low); not all warnings are exploitable. The chapter's `.claude/hooks/verify.sh` should be configured to fail only on High-confidence issues (or configurable threshold).
- False positives can occur in complex control flow; the chapter's `.rubocop.yml` / `.brakeman.yml` should document which checks are configured or suppressed (and why).
- Date checked: **2026-09-04**.

**Recommendation:**

1. **Wire Brakeman into `.claude/hooks/verify.sh`** as a gate before merge: `brakeman -q --no-summary`. Fail on any High-confidence warning.
2. **Configure `.brakeman.yml`** to document exceptions (e.g., intentional use of `.html_safe` in a sanitized context) and reasoning, so reviewers and readers understand the security trade-offs.
3. **Teach the three checks above (SQL injection, mass assignment, XSS)** in the chapter's governance narrative. Show a realistic Brakeman output (from running the example Rails app) in the validation artefact (`artefacts/rails-validation-log.md`).
4. **Do not suppress checks lightly.** If Brakeman flags something, investigate and either fix the code or document the suppression in `.brakeman.yml` with a comment explaining why it is safe.

This makes Brakeman a visible, enforceable gate for the reader, demonstrating how a static scanner prevents entire categories of vulnerabilities from being merged.
