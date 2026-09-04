# NOTE-SDLC-4-1: Ruby and Rails versions, plus gate gem versions

**Answer:**
Current stable Ruby version is **4.0.6** (released July 14, 2026). Current stable Rails version is **8.1.3.1** (released July 29, 2026). Gate gems: **rspec-rails 8.0.4** (March 11, 2026), **rubocop 1.86.0** (March 23, 2026), **rubocop-rails 2.37.0**, **brakeman 8.0.6** (August 2026), **bcrypt 3.1.22** (March 18, 2026, includes CVE-2026-33306 fix).

**Evidence:**

1. **Ruby 4.0.6** — [Ruby 4.0.6 Released](https://www.ruby-lang.org/en/news/2026/07/14/ruby-4-0-6-released/) (July 14, 2026)

2. **Rails 8.1.3.1** — [Rails Versions 8.0.5 and 8.1.3 have been released](https://rubyonrails.org/2026/3/24/Rails-Versions-8-0-5-and-8-1-3-have-been-released) (latest stable, dated March 24, 2026 for 8.1.3; 8.1.3.1 is a security patch released July 29, 2026 per official release notes). See also [All versions of rails on RubyGems](https://rubygems.org/gems/rails/versions).

3. **rspec-rails 8.0.4** — [All versions of rspec-rails on RubyGems](https://rubygems.org/gems/rspec-rails/versions) shows 8.0.4 released March 11, 2026. See also [rspec-rails on RubyGems](https://rubygems.org/gems/rspec-rails).

4. **rubocop 1.86.0** — [RuboCop releases on GitHub](https://github.com/rubocop/rubocop/releases) shows 1.86.0 as latest stable (March 23, 2026). Confirmed via [All RuboCop versions on RubyGems](https://rubygems.org/gems/rubocop/versions).

5. **rubocop-rails 2.37.0** — [All versions of rubocop-rails on RubyGems](https://rubygems.org/gems/rubocop-rails/versions) shows 2.37.0 as current.

6. **brakeman 8.0.6** — [Brakeman Security Scanner](https://brakemanscanner.org/) official site; 8.0.6 released August 2026. Confirmed via gem releases and documentation stating "requires at least Ruby 3.2.0 to run" and "ships 86 checks covering SQL injection, XSS, command injection, CSRF, mass assignment, path traversal, unsafe deserialization, and remote code execution."

7. **bcrypt 3.1.22** — [bcrypt on RubyGems](https://rubygems.org/gems/bcrypt) shows 3.1.22 as current; released March 18, 2026 with fix for CVE-2026-33306 (integer overflow in Java extension). See [bcrypt-ruby GitHub CHANGELOG](https://github.com/bcrypt-ruby/bcrypt-ruby/blob/master/CHANGELOG).

**Caveats / limits:**

- Rails 8.1 series will continue receiving bugfixes until October 2026; after that, only security updates. Rails 8.0 series shifted to security-only updates in May 2026. Pin the exact minor version (8.1.3.1) in the Gemfile to avoid drift during the chapter's lifecycle.
- Brakeman requires Ruby ≥ 3.2.0; no issue with Ruby 4.0.6.
- All dates checked **2026-09-04**.

**Recommendation:**

Pin exact versions in the example Rails project's `Gemfile`:
```ruby
ruby "4.0.6"
gem "rails", "8.1.3.1"
gem "rspec-rails", "8.0.4"
gem "rubocop", "1.86.0"
gem "rubocop-rails", "2.37.0"
gem "brakeman", "8.0.6"
gem "bcrypt", "3.1.22"
```

This ensures reproducibility for the reader running the example in a declared Rails 8.1 environment outside the Python book repository.
