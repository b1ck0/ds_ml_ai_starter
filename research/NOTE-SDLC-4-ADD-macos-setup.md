# NOTE-SDLC-4-ADD-macos-setup.md

macOS local-environment setup for Rails 8 e-store (SPEC-SDLC-4 + SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents).
All facts verified against live, authoritative sources. Date checked: **2026-09-04**.

---

## 1. Installing Ruby on macOS via Homebrew and rbenv

### Answer
**Homebrew** is the macOS package manager; **rbenv** + **ruby-build** (installed via Homebrew) is the current standard version manager for Ruby; **asdf** is a newer multi-language alternative. Install Ruby 4.0.6 using `rbenv install 4.0.6` and `rbenv global 4.0.6`. Initialize rbenv in `~/.zshrc` with `eval "$(rbenv init - zsh)"`.

### Evidence
- **Homebrew install** (pre-requisite): [Homebrew Documentation: Installation](https://docs.brew.sh/Installation)
- **rbenv + ruby-build via Homebrew**: [rbenv GitHub](https://github.com/rbenv/rbenv), [ruby-build GitHub](https://github.com/rbenv/ruby-build), [rbenv official site](https://rbenv.org/), [Homebrew Formulae: rbenv](https://formulae.brew.sh/formula/rbenv)
  - Install: `brew install rbenv ruby-build`
  - Install Ruby version: `rbenv install 4.0.6` (exact command per [rbenv usage](https://github.com/rbenv/rbenv#basic-github-checkout))
  - Set global version: `rbenv global 4.0.6`
- **Ruby 4.0.6 release date**: July 14, 2026 ([Ruby 4.0.6 Released](https://www.ruby-lang.org/en/news/2026/07/14/ruby-4-0-6-released/))
- **rbenv ~/.zshrc initialization**: `export PATH="$HOME/.rbenv/bin:$PATH"` and `eval "$(rbenv init - zsh)"` (note: `-zsh` flag required for zsh, not bash). [rbenv zsh setup issues](https://github.com/rbenv/rbenv/issues/1631), [Better Stack: Getting Started with rbenv](https://betterstack.com/community/guides/scaling-ruby/rbenv-explained/)
- **asdf alternative**: Install via `brew install asdf`, then add to `~/.zprofile` or `~/.zshrc`. Uses ruby-build plugin internally. [SitePoint: Ruby Version Managers for macOS](https://www.sitepoint.com/ruby-version-managers-macos/), [Mac Install Guide: Compare Ruby Version Managers](https://mac.install.guide/ruby/5), [asdf in macOS blog](https://www.jdeen.com/blog/installing-ruby-using-asdf-in-macos)
- **Official Rails guide note**: As of Sept 2026, the official [Rails installation guide](https://guides.rubyonrails.org/install_ruby_on_rails.html) now recommends **Mise** (not rbenv) as the preferred version manager, but rbenv remains a working, widely-documented alternative.

### Caveats / Limits
- The official Rails guide has shifted toward Mise in 2026; rbenv/ruby-build is still fully functional but no longer the default recommendation.
- Ensure `~/.zshrc` has the exact `eval "$(rbenv init - zsh)"` line (not `rbenv init -` without the `zsh` argument, which is for bash).
- If .zshrc already has a commented-out rbenv eval line, rbenv's init may incorrectly skip initialization; remove the comment.

### Recommendation
**For quick start**: Follow the steps exactly; add the Homebrew install and rbenv init commands to the chapter's setup section. **Note for readers**: Mention that Mise is now the official default (per 2026 Rails guide), but rbenv/ruby-build is still a solid choice if already familiar. If writing a "fast path," suggest using the system Homebrew + rbenv (no configuration needed beyond installing and sourcing ~/.zshrc).

---

## 2. Installing Rails 8.1.3.1 and Bundler

### Answer
Install Rails 8.1.3.1 with `gem install rails -v 8.1.3.1`. Bundler is installed automatically with Ruby via rbenv; run `bundle install` in the project directory to resolve and lock gem dependencies from the Gemfile.

### Evidence
- **Rails 8.1.3.1 release date**: July 29, 2026 ([Rails 7.2.3.2, 8.0.5.1, and 8.1.3.1 released](https://rubyonrails.org/2026/7/29/Rails-Versions-7-2-3-2-8-0-5-1-and-8-1-3-1-have-been-released), [RubyGems rails page](https://rubygems.org/gems/rails/versions/8.1.3.1))
- **gem install syntax**: `gem install rails -v 8.1.3.1` (or `gem install rails --version 8.1.3.1`). Long form per [RubyGems install guide](https://web.stanford.edu/~ouster/cgi-bin/cs142-fall10/railsInstall.php), [til: select-a-specific-rails-version-to-install](https://github.com/jbranchaud/til/blob/master/rails/select-a-specific-rails-version-to-install.md)
- **Bundler**: Installed automatically with Ruby; included in stdlib since Ruby 2.6+. Official docs: [Bundler: bundle install](https://bundler.io/v2.4/man/bundle-install.1.html), [RubyGems Guides: bundle install](https://guides.rubygems.org/command-reference/bundle-install/)
  - Command: `bundle install` (scans Gemfile, resolves dependencies, creates Gemfile.lock, installs to system gem location or vendor/bundle if --deployment used)

### Caveats / Limits
- Rails 8.1.3.1 includes a security fix (CVE-2026-66066) for Active Storage variant processing; ensure this version is used in production notes.
- Bundler comes with Ruby; no separate install needed if using rbenv's Ruby.

### Recommendation
Pin Rails to 8.1.3.1 in the Gemfile: `gem 'rails', '8.1.3.1'`. Then `bundle install` to generate Gemfile.lock. Include the `gem install rails -v 8.1.3.1` command in the setup section if testing Rails CLI outside a project context.

---

## 3. Database Options: SQLite vs PostgreSQL on macOS

### Answer
**SQLite** is built into macOS and the Rails default; use it for zero-config local testing. **PostgreSQL** is installed via `brew install postgresql@16` (or current major version); the `pg` gem requires `brew install libpq` + `bundle config build.pg --with-pg-config=/opt/homebrew/opt/libpq/bin/pg_config` on Apple Silicon Macs.

### Evidence

#### SQLite
- **macOS includes SQLite**: Pre-installed at `/usr/bin/sqlite3`. [SQLite on macOS: Not ACID compliant](https://bonsaidb.io/blog/acid-on-apple/), [List of built-in macOS apps (Wikipedia)](https://en.wikipedia.org/wiki/List_of_built-in_macOS_apps), [How to install SQLite3 on macOS](https://sqldocs.org/sqlite3-on-macos/)
- **sqlite3 gem**: Supports precompiled binaries for arm64-darwin (Apple Silicon). Latest version (as of Sept 2026) is 2.7.0. [sqlite3 gem on RubyGems](https://rubygems.org/gems/sqlite3/versions/2.7.0), [sqlite3-ruby INSTALLATION.md](https://github.com/sparklemotion/sqlite3-ruby/blob/main/INSTALLATION.md)
  - No extra system dependencies beyond Xcode Command Line Tools (for gem compilation).
- **Rails database.yml (SQLite)**:
  ```yaml
  default: &default
    adapter: sqlite3
    pool: 5
    timeout: 5000
  
  development:
    <<: *default
    database: db/development.sqlite3
  
  test:
    <<: *default
    database: db/test.sqlite3
  
  production:
    <<: *default
    database: db/production.sqlite3
  ```
  Source: [Rails Database.yml examples GitHub gist](https://gist.github.com/datt/e12fa0da294e7a8f3ac96abee346a098), [Sample config/database.yml GitHub gist](https://gist.github.com/jwo/4512764)

#### PostgreSQL
- **Homebrew install**: `brew install postgresql@16` (or `brew install postgresql@<current_major_version>`). Current stable major versions: 16, 17, 18 available. [Homebrew Formulae: postgresql@16](https://formulae.brew.sh/formula/postgresql@16), [PostgreSQL: macOS packages](https://www.postgresql.org/download/macosx/)
  - Start service: `brew services start postgresql@16`
  - Apple Silicon note: Data directory defaults to `/opt/homebrew/var/postgresql@16` (Intel uses `/usr/local`)
- **libpq for pg gem** (required for `pg` Ruby gem):
  - Install: `brew install libpq`
  - Configure bundler (Apple Silicon): `bundle config build.pg --with-pg-config=/opt/homebrew/opt/libpq/bin/pg_config`
  - Configure bundler (Intel): `bundle config build.pg --with-pg-config=/usr/local/opt/libpq/bin/pg_config`
  - Then: `bundle install`
  - Source: [Install postgresql gem `pg` on macOS (GitHub gist)](https://gist.github.com/tomholford/f38b85e2f06b3ddb9b4593e841c77c9e), [Failed install gem 'pg' on macos (Medium)](https://ervinismu.medium.com/failed-install-gem-pg-on-macos-9a2dc102cae9), [OSX Can't find libpq-fe.h (Medium)](https://medium.com/@ssscripting/osx-cant-find-the-libpq-fe-h-header-e41b5a25041c)
- **Rails database.yml (PostgreSQL)**:
  ```yaml
  default: &default
    adapter: postgresql
    encoding: unicode
    pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>
  
  development:
    <<: *default
    database: myapp_development
  
  test:
    <<: *default
    database: myapp_test
  
  production:
    <<: *default
    database: myapp_production
    username: <%= ENV['DB_USER'] %>
    password: <%= ENV['DB_PASSWORD'] %>
  ```
  Source: [Rails Database.yml examples GitHub gist](https://gist.github.com/datt/e12fa0da294e7a8f3ac96abee346a098), [Setting Up and Managing Databases in Rails (Medium)](https://bhartee-tech-ror.medium.com/setting-up-and-managing-databases-in-ruby-on-rails-72885f9f1164)

### Caveats / Limits
- SQLite is suitable for development and small projects; not recommended for multi-process Rails servers (e.g., Puma in production) due to locking issues.
- PostgreSQL installation on Homebrew is "keg-only," meaning binaries are not in PATH; libpq configuration is mandatory.
- The exact PostgreSQL@N formula available in Homebrew can change; check `brew info postgresql` for current version.
- Apple Silicon (arm64) uses `/opt/homebrew` prefix; Intel uses `/usr/local`. Bundle config must point to the correct path.

### Recommendation
**For the quick example**: Use SQLite (built-in, zero setup). The database.yml is already correct in new Rails 8 apps (`bin/rails db:setup` will create it). **For production-like testing**: Use PostgreSQL with the full libpq+pg gem setup. Include a note: "If you're on Apple Silicon, use `/opt/homebrew/opt/libpq`; if Intel, use `/usr/local/opt/libpq`."

---

## 4. Node.js LTS on macOS

### Answer
**Node.js 24 LTS** is the current long-term support version as of September 2026. Install via `brew install node`. Node.js 26 will enter LTS in October 2026 (currently "Current" status).

### Evidence
- **Node.js 24 LTS (Sept 2026)**: As of June 2026, Node 24 is the active LTS line. [Node.js on endoflife.date](https://endoflife.date/nodejs), [Node.js Moves to One Major Release Per Year (InfoQ)](https://www.infoq.com/news/2026/06/nodejs-release-changes/), [Node.js Releases](https://nodejs.org/en/about/previous-releases)
- **Node.js 26 status**: Will enter LTS in October 2026; currently "Current" release (Sept 2026). [Node.js 26.0.0 announcement](https://nodejs.org/en/blog/release/v26.0.0/), [Node 22 vs Node 24 in 2026 guide](https://www.pkgpulse.com/guides/nodejs-22-vs-nodejs-24-2026)
- **Homebrew install**: `brew install node` (installs current Node version; to pin to LTS explicitly, use `brew install node@24` or similar).

### Caveats / Limits
- Node.js 26 will become LTS in October 2026; after that date, recommend Node 26 LTS instead.
- Homebrew's `node` formula installs the latest stable version; for LTS certainty, use `brew install node@24` to pin major version.

### Recommendation
Pin Node.js 24 LTS for the Rails 8 e-store project (used for Lighthouse CI `@lhci/cli` and frontend tooling). Include the command `brew install node@24` or note that `brew install node` is acceptable if the latest stable LTS is already 24+.

---

## 5. macOS Gotchas and Prerequisites

### Answer
**Xcode Command Line Tools** (`xcode-select --install`) are **required** for native gem compilation. **Apple Silicon Macs** use Homebrew prefix `/opt/homebrew` (not `/usr/local`). Key gotchas: `pg` and `nokogiri` gems need native compilation; libpq and Xcode tools are mandatory.

### Evidence

#### Xcode Command Line Tools
- **Requirement**: All Ruby development on macOS requires Xcode Command Line Tools because gems with native extensions (C/C++) must be compiled locally.
- **What it provides**: C compiler (clang/gcc), make, system headers, SDKs, and CommonCrypto/OpenSSL alternatives.
- **Install command**: `xcode-select --install` (opens System Software Update dialog).
- **Size**: ~500 MB (much smaller than full Xcode app, which is 40+ GB).
- Source: [Xcode Command Line Tools (Ruby on Rails Install Guide)](https://mac.install.guide/rubyonrails/2), [Ruby (Mac Install Guide)](https://mac.install.guide/ruby/2), [GitHub issue: Apple XCode required before installing Ruby](https://github.com/rvm/rvm/issues/1632), [Ruby on Rails install guide](https://www.moncefbelyamani.com/how-to-install-rails-and-create-a-new-rails-app-on-a-mac-the-easy-way/)

#### Apple Silicon (arm64) and Homebrew Prefix
- **Default prefix**: `/opt/homebrew` (ARM64 native); Intel Macs use `/usr/local`.
- **Reason**: Apple Silicon can run Intel apps via Rosetta 2; keeping separate prefixes avoids binary conflicts.
- **Impact**: libpq, PostgreSQL, and other Homebrew-installed tools are at `/opt/homebrew/opt/<formula>` on Apple Silicon.
- **In shell config**: If adding Homebrew to PATH on Apple Silicon, use `/opt/homebrew/bin` (not `/usr/local/bin`).
- Source: [Homebrew Documentation: Installation](https://docs.brew.sh/Installation), [Homebrew on Apple Silicon Macs (Andre Arko blog)](https://andre.arko.net/2021/02/11/homebrew-on-apple-silicon-macs/), [Troubleshooting Homebrew Mixed Installations on Apple Silicon (Zenn)](https://zenn.dev/ryok/articles/homebrew-arm64-x86-mixed-fix?locale=en), [Apple Silicon Tips for Engineers (Substack)](https://strlen.substack.com/p/apple-silicon-tips-for-engineers)

#### Common Gem Gotchas
- **pg gem**: Requires libpq; see Section 3 for full setup.
- **nokogiri gem**: Requires native compilation; Xcode tools mandatory.
- **JSON, bcrypt**: Both require native C compilation.

### Caveats / Limits
- If Xcode Command Line Tools are not installed, `rbenv install` or `bundle install` will fail on gems with native extensions.
- Mixing Intel and ARM64 Homebrew installations can cause conflicts; stick to one architecture per machine.

### Recommendation
**Add to the chapter preamble**: "First, run `xcode-select --install` to install Xcode Command Line Tools (one-time step)." Then confirm it's installed before proceeding. For Apple Silicon readers, explicitly note the `/opt/homebrew` prefix when setting up libpq or PostgreSQL.

---

## 6. Standard Rails 8 Run and Verification Commands

### Answer
- **Database setup**: `bin/rails db:setup` (create database, load schema, seed), `bin/rails db:migrate` (apply migrations), `bin/rails db:seed` (run seeds.rb only).
- **Development server**: `bin/rails server` (or alias `bin/rails s`; default port 3000).
- **Testing & linting**: `bundle exec rspec` (unit tests), `bundle exec rubocop` (code style), `bundle exec brakeman` (security scanner).

### Evidence

#### Database Commands
- **bin/rails db:setup**: Creates database, loads schema, initializes with seed data. [Active Record Migrations Rails Guides](https://guides.rubyonrails.org/active_record_migrations.html), [Complete Guide to Rails Database Commands (Rails Drop)](https://railsdrop.com/2025/08/27/the-complete-guide-to-rails-database-commands-from-basics-to-production/)
- **bin/rails db:migrate**: Applies pending migrations to update the schema.
- **bin/rails db:seed**: Runs `db/seeds.rb` to populate initial data.
- **bin/rails db:prepare** (alternative): Idempotent version of db:setup; safe to run multiple times.
- Source: [Rails migration guide](https://guides.rubyonrails.org/active_record_migrations.html), [DEV: All the database tasks in Rails](https://dev.to/software_writer/all-the-database-tasks-in-rails-2jic), [Linyclar: bin/rails db](https://linyclar.github.io/rails_memos/commands/db/)

#### Development Server
- **bin/rails server** (or `bin/rails s`): Starts WEBrick or Puma development server on localhost:3000 by default.
- **Port override**: `bin/rails server -p 4000` (to run on port 4000).
- Source: [Rails Command Line Guide](https://guides.rubyonrails.org/command_line.html)

#### Testing and Linting
- **bundle exec rspec**: Runs RSpec test suite (unit/integration tests). Not a Rails native command; requires `rspec-rails` gem in Gemfile.
- **bundle exec rubocop**: Runs RuboCop static code analyzer (style/lint checker). Not native Rails; requires `rubocop` gem.
- **bundle exec brakeman**: Runs Brakeman security scanner (Rails-specific). Requires `brakeman` gem.
- Source: [Git hooks with Brakeman, Rubocop, Rspec (GitHub)](https://gist.github.com/fsdevblog/12ceacc5de9239628710b4085cc82057), [DevOps with Rails: CI with Github Actions and Lefthook](https://eagerworks.com/blog/devops-with-rails-setting-up-ci-github-actions-and-lefthook), community practice (these are not official Rails commands but standard Rails development tooling)

### Caveats / Limits
- `bin/rails db:setup` will fail if the database already exists; use `bin/rails db:reset` to drop and recreate.
- RSpec, RuboCop, and Brakeman are external gems (not Rails native); they must be in the Gemfile and available via bundler.
- The exact command syntax may vary slightly if these gems are configured with custom options (e.g., `rubocop --parallel`, `brakeman -q`).

### Recommendation
Include all six commands in the chapter's verification section. Emphasize that `bin/rails db:setup` is the one-time setup command; subsequent development uses `bin/rails db:migrate`. Recommend testing the Rails server immediately after setup: `bin/rails server` and visit `http://localhost:3000` in the browser to confirm it starts.

---

## 7. Lighthouse CI and npm Tooling

### Answer
**Lighthouse CI** (`@lhci/cli`) is an npm package for automated performance auditing in CI/CD pipelines. Install via `npm install @lhci/cli` (or npm i -g @lhci/cli for global install), then run `npx lhci autorun` to execute performance audits.

### Evidence
- **@lhci/cli npm package**: [npm: @lhci/cli](https://www.npmjs.com/package/@lhci/cli)
- **Installation**: `npm install @lhci/cli` (local to project) or `npm install -g @lhci/cli` (global system install). For CI pipelines, the pattern is `npm install -g @lhci/cli` followed by `npx lhci autorun`.
- **Official docs**: [GitHub: GoogleChrome/lighthouse-ci](https://github.com/GoogleChrome/lighthouse-ci/), [lighthouse-ci getting started](https://github.com/GoogleChrome/lighthouse-ci/blob/main/docs/getting-started.md), [Lighthouse CI docs](https://googlechrome.github.io/lighthouse-ci/docs/getting-started.html)
- **Web.dev reference**: [Performance monitoring with Lighthouse CI (web.dev)](https://web.dev/articles/lighthouse-ci)

### Caveats / Limits
- Lighthouse CI is a CI/CD tool; not required for local Rails development (use only if automating performance checks per the QA-agents spec).
- Requires Node.js to be installed (see Section 4).
- Lighthouse CI requires configuration file (`.lighthouserc.json`) to specify audit targets and thresholds; this is **not** part of macOS setup but part of SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents.

### Recommendation
**For SPEC-SDLC-4-ADDENDUM only**: Include npm installation (Node.js 24 LTS) and `npm install @lhci/cli` as a separate optional step in the QA section. Do **not** include in the main Rails 8 e-store setup section unless the spec explicitly requires it. Document that `npx lhci autorun` is run as part of CI/CD automation, not local development.

---

## Summary: Verified Command Checklist for macOS Rails 8 Setup

All commands verified against live authoritative sources (date checked: **2026-09-04**).

| Step | Command | Source |
|------|---------|--------|
| 1. Xcode Tools | `xcode-select --install` | [Mac Install Guide](https://mac.install.guide/ruby/2) |
| 2. Homebrew | `brew install rbenv ruby-build` | [Homebrew rbenv formula](https://formulae.brew.sh/formula/rbenv) |
| 3. Init .zshrc | `eval "$(rbenv init - zsh)"` | [rbenv GitHub](https://github.com/rbenv/rbenv/issues/1631) |
| 4. Install Ruby | `rbenv install 4.0.6 && rbenv global 4.0.6` | [Ruby 4.0.6 release](https://www.ruby-lang.org/en/news/2026/07/14/ruby-4-0-6-released/) |
| 5. Install Rails | `gem install rails -v 8.1.3.1` | [RubyGems rails versions](https://rubygems.org/gems/rails/versions/8.1.3.1) |
| 6. Bundle install | `bundle install` | [Bundler docs](https://bundler.io/v2.4/man/bundle-install.1.html) |
| 7a. Setup DB (SQLite) | `bin/rails db:setup` | [Rails Guides: Active Record Migrations](https://guides.rubyonrails.org/active_record_migrations.html) |
| 7b. Start server | `bin/rails server` | [Rails Guides: Command Line](https://guides.rubyonrails.org/command_line.html) |
| 8. Tests | `bundle exec rspec` | Community standard (gems: rspec-rails) |
| 9. Lint | `bundle exec rubocop` | Community standard (gem: rubocop) |
| 10. Security | `bundle exec brakeman` | Community standard (gem: brakeman) |
| Optional: Node LTS | `brew install node@24` | [endoflife.date nodejs](https://endoflife.date/nodejs) |
| Optional: Lighthouse CI | `npm install @lhci/cli` | [npm @lhci/cli](https://www.npmjs.com/package/@lhci/cli) |

---

## Sources (All URLs checked 2026-09-04)

### Official Project/Framework Documentation
- [Ruby 4.0.6 Released](https://www.ruby-lang.org/en/news/2026/07/14/ruby-4-0-6-released/)
- [Rails 7.2.3.2, 8.0.5.1, and 8.1.3.1 released](https://rubyonrails.org/2026/7/29/Rails-Versions-7-2-3-2-8-0-5-1-and-8-1-3-1-have-been-released)
- [Ruby on Rails Guides: Getting Started](https://guides.rubyonrails.org/install_ruby_on_rails.html)
- [Rails Guides: Active Record Migrations](https://guides.rubyonrails.org/active_record_migrations.html)
- [Rails Guides: Command Line](https://guides.rubyonrails.org/command_line.html)
- [Bundler: bundle install](https://bundler.io/v2.4/man/bundle-install.1.html)
- [RubyGems Guides: bundle install](https://guides.rubygems.org/command-reference/bundle-install/)

### Homebrew & Version Managers
- [Homebrew Documentation: Installation](https://docs.brew.sh/Installation)
- [Homebrew Formulae: rbenv](https://formulae.brew.sh/formula/rbenv)
- [Homebrew Formulae: postgresql@16](https://formulae.brew.sh/formula/postgresql@16)
- [rbenv GitHub](https://github.com/rbenv/rbenv)
- [ruby-build GitHub](https://github.com/rbenv/ruby-build)
- [rbenv official site](https://rbenv.org/)
- [Homebrew on Apple Silicon Macs](https://andre.arko.net/2021/02/11/homebrew-on-apple-silicon-macs/)

### Package/Gem Information
- [RubyGems: rails versions](https://rubygems.org/gems/rails/versions/8.1.3.1)
- [RubyGems: sqlite3 gem](https://rubygems.org/gems/sqlite3/versions/2.7.0)
- [sqlite3-ruby INSTALLATION.md](https://github.com/sparklemotion/sqlite3-ruby/blob/main/INSTALLATION.md)
- [npm: @lhci/cli](https://www.npmjs.com/package/@lhci/cli)
- [GoogleChrome/lighthouse-ci GitHub](https://github.com/GoogleChrome/lighthouse-ci/)

### Node.js & Build Tools
- [Node.js on endoflife.date](https://endoflife.date/nodejs)
- [Node.js Moves to One Major Release Per Year](https://www.infoq.com/news/2026/06/nodejs-release-changes/)

### macOS-Specific Guidance
- [Mac Install Guide: Xcode Command Line Tools](https://mac.install.guide/ruby/2)
- [Mac Install Guide: Compare Ruby Version Managers](https://mac.install.guide/ruby/5)
- [Troubleshooting Homebrew Mixed Installations on Apple Silicon](https://zenn.dev/ryok/articles/homebrew-arm64-x86-mixed-fix?locale=en)

### Database Configuration
- [Rails Database.yml examples (GitHub gist)](https://gist.github.com/datt/e12fa0da294e7a8f3ac96abee346a098)
- [SQLite on macOS: Not ACID compliant](https://bonsaidb.io/blog/acid-on-apple/)
- [PostgreSQL: macOS packages](https://www.postgresql.org/download/macosx/)
- [Install postgresql gem pg on macOS (GitHub gist)](https://gist.github.com/tomholford/f38b85e2f06b3ddb9b4593e841c77c9e)

### Related Guides & Blogs
- [SitePoint: Ruby Version Managers for macOS](https://www.sitepoint.com/ruby-version-managers-macos/)
- [Installing Ruby using ASDF in macOS](https://www.jdeen.com/blog/installing-ruby-using-asdf-in-macos)
- [Better Stack: Getting Started with rbenv](https://betterstack.com/community/guides/scaling-ruby/rbenv-explained/)
- [Setting Up and Managing Databases in Rails (Medium)](https://bhartee-tech-ror.medium.com/setting-up-and-managing-databases-in-ruby-on-rails-72885f9f1164)

