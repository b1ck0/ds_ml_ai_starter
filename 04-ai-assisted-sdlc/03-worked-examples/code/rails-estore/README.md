# rails-estore

A small Rails e-store — sign-up, login, a cart, checkout with a stubbed Stripe seam, and a public
product catalog — built under a governed, spec-driven SDLC. This README is a **standalone, zero-to-
running guide for macOS**: everything you need is here, whether or not you've read the book chapter
this project ships as a worked example for
([`04-ai-assisted-sdlc/03-worked-examples/03-rails-estore-sdlc.md`](../../03-rails-estore-sdlc.md) —
that chapter explains the *governance* story: the `.claude/` scaffold, the spec → ground → implement
→ gate → review → merge loop, and a real authorization bug a fresh reviewer caught that three
automated gates missed. This file is the "make it actually run on my Mac" companion to that story.

Every command below is grounded and dated 2026-09-04
(`docs/research/NOTE-SDLC-4-ADD-macos-setup.md`, `docs/research/NOTE-SDLC-4-ADD-1-gem-npm-versions.md`)
— re-check current versions before you rely on this for a new project; software moves.

## What you get

![The rails-estore product catalog: a "Shop all products" heading, a search box, and four seeded products — Rails Mug $15.00, Convention Over Configuration T-Shirt $25.00, Omakase Sticker Pack $8.00, and Migration Notebook $12.00 — each with a placeholder image and a link.](docs/screenshots/storefront.png)

The product catalog at `/products`, captured **live from this project running under `docker compose up`**
(Docker 28.4.0 / Compose v2.39.2). A product page, with the authorization-aware "Sign in to buy"
prompt because the visitor is not signed in:

![The Rails Mug product page: the product name, a placeholder image, the description "A ceramic mug for Rubyists. Holds 350ml of coffee and zero N+1 queries.", the price $15.00, and a "Sign in to buy" link.](docs/screenshots/product.png)

> These pages ship **no CSS on purpose** — this is a teaching scaffold where clean, semantic,
> accessible HTML is the point (exactly what the `frontend-qa` and `seo-optimizer` agents check).
> Styling is a natural first "vibe-engineering" task once you're set up. The sign-in page lives at
> `/session/new` (`docs/screenshots/login.png`).

## New to Docker? Start here

If you've never used Docker, two ideas cover everything you need for this project:

- A **container** is a self-contained box that carries the app AND its exact Ruby/Rails/SQLite
  versions with it — so "works on my machine" stops being a problem, because your machine only runs
  the box, not the app directly. Think of it as a JAR that packages its own JVM, not just its own
  classes.
- An **image** is the read-only blueprint a container is started from (built once by `docker build`,
  from the instructions in this project's `Dockerfile`); a **container** is a running instance of
  that image, the way an object is an instance of a class.
- **Docker Compose** is the layer above a single container: one `docker-compose.yml` file describes
  the service(s) an app needs (here, just one: `web`) and how they're wired together (which port maps
  to which, where the database file persists), so one command starts the whole thing instead of a
  long `docker run` invocation with a dozen flags.

**Installing Docker Desktop on macOS** — download the `.dmg` for your Mac's chip (Apple Silicon or
Intel) from [docs.docker.com/desktop/setup/install/mac-install](https://docs.docker.com/desktop/setup/install/mac-install/)
(checked 2026-09-04), open it, and drag the Docker icon into Applications; Docker Desktop is
supported on the current macOS release and the two before it. A Homebrew cask (`brew install --cask
docker`) is a common alternative if you already manage everything else through Homebrew, though the
`.dmg` download is what Docker's own docs walk through. Either way, **launch Docker.app once** after
installing — Compose talks to a background daemon that only starts when the app is running (look for
the whale icon in the menu bar).

The handful of commands you'll actually use, each doing one thing:

| Command | What it does |
|---|---|
| `docker compose up` | Builds the image (first time only) and starts the app in your terminal, logs streaming live. Add `-d` to run it in the background instead. |
| `docker compose ps` | Shows whether the `web` service is up and which port it's mapped to. |
| `docker compose logs -f` | Tails the running container's logs — useful if you started with `-d`. |
| `docker compose down` | Stops and removes the container. Add `-v` to also delete the named volume holding the SQLite database (a real reset, not just a restart). |

Then open **<http://localhost:3000>** in a browser — that's it.

**Why this is the easygoing path:** nothing in §1 below (Xcode Command Line Tools, Homebrew, rbenv,
a matching Ruby, `gem install rails`) is needed when you run this way. Docker carries all of it
inside the image; your Mac only needs Docker itself.

## Quickest start — Docker Compose

```bash
docker compose up
```

Then open <http://localhost:3000> — the product catalog, seeded with four sample products, is what
you'll see. This is the path actually built, run, and verified for this addendum — real `docker
compose build`, `docker compose up`, `curl`, and `bundle exec rspec` output, captured against
**Docker 28.4.0 / Docker Compose v2.39.2** on the authoring machine, is in
[`artefacts/rails-validation-log.md`](../../artefacts/rails-validation-log.md)'s Docker section —
unlike the rest of this Rails example (correct-but-not-executed in the book's own repo, since no
Ruby toolchain runs there), this is the one part that was genuinely exercised end-to-end.

Everything below — native macOS setup, the individual gates, the frontend gate, project layout,
troubleshooting — is the alternative path for when you want to run Ruby directly on your Mac (to use
a debugger, an editor's inline test runner, etc.) rather than inside a container.

## 1. Prerequisites

| Tool | Why you need it | Install |
|---|---|---|
| Xcode Command Line Tools | Compiles native-extension gems (`sqlite3`, `bcrypt`, `nokogiri` if pulled in transitively) — without it `bundle install` fails on the first gem with C code. | `xcode-select --install` |
| Homebrew | The package manager everything else below comes through. | See [brew.sh](https://brew.sh) / [Homebrew install docs](https://docs.brew.sh/Installation) |
| rbenv + ruby-build | Installs and switches Ruby versions per-project — the Ruby equivalent of a Java version manager (sdkman/jenv). | `brew install rbenv ruby-build` |
| Ruby 4.0.6 | The exact version this Gemfile pins (`ruby "4.0.6"`). | `rbenv install 4.0.6 && rbenv global 4.0.6` |
| Rails 8.1.3.1 | This project's pinned Rails version — installed as a gem once Ruby is set up. | `gem install rails -v 8.1.3.1` |
| Node.js 24 LTS | Only needed for the **frontend gate**'s Lighthouse CI step (§6) — not required to run the app itself. | `brew install node@24` |

**Apple Silicon note:** Homebrew installs under `/opt/homebrew` (not `/usr/local`, which is the Intel
prefix). If you ever add PostgreSQL or another compiled dependency, point `bundle config` at
`/opt/homebrew/opt/<formula>` — see Troubleshooting below. This project uses **SQLite**, which needs
no such configuration at all.

### 1a. Initialize rbenv in your shell

Add this to `~/.zshrc` (macOS's default shell is zsh; note the `zsh` argument — `rbenv init -` alone
is the bash form and silently does the wrong thing under zsh):

```bash
export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init - zsh)"
```

Then reload your shell (`source ~/.zshrc` or open a new terminal tab) and confirm:

```bash
rbenv versions
# should list 4.0.6 once you've run `rbenv install 4.0.6` above
```

## 2. Get the project running

```bash
# From this directory (code/rails-estore/):
bundle install                       # resolves and installs every gem in the Gemfile
cp .env.example .env                 # copy the config template — fill in TEST-mode Stripe keys if
                                      # you want to exercise the real Stripe::Checkout::Session path;
                                      # STRIPE_STUB=true (the default) needs no Stripe account at all
bin/rails db:setup                   # creates db/development.sqlite3 + db/test.sqlite3, loads
                                      # db/schema.rb, and runs db/seeds.rb (a handful of sample
                                      # products so the catalog isn't empty on first load)
bin/rails server                     # starts the app at http://localhost:3000
```

Open <http://localhost:3000> — you should see the product catalog with four seeded products (a mug,
a T-shirt, a sticker pack, a notebook). Sign up at `/registration`, add something to your cart, and
check out (checkout uses `PaymentService`'s stubbed path by default — no real Stripe account, no real
charge, see `app/services/payment_service.rb`).

**Why SQLite:** it's the Rails 8 default, built into macOS, and needs zero server setup — the right
choice for a friend's first run of this project. `config/database.yml` is already configured for it;
nothing else to install. (If you later want PostgreSQL for something closer to a production setup,
`docs/research/NOTE-SDLC-4-ADD-macos-setup.md` §3 has the full `brew install postgresql@16` +
`libpq` + `bundle config build.pg` sequence — out of scope for this quick-start.)

## 3. Run every gate

Each of these is what `.claude/hooks/verify.sh` runs automatically on every file edit when you're
driving this project through Claude Code — running them by hand here is the same checks, on demand.

| Command | What it's for |
|---|---|
| `bundle exec rspec` | The full test suite — every acceptance criterion from `docs/features/FEATURE-*.md` has a passing example here. |
| `bundle exec rubocop` | Style/lint — also enforces this project's "controllers stay thin" rule (`Metrics/MethodLength: 15`), because a controller action too long to read in one glance is one a reviewer can't authorization-check by eye. |
| `bundle exec brakeman -q --no-summary` | Static security scan — mass assignment, SQL injection, XSS, and more. Must report zero High-confidence warnings. |

```bash
bundle exec rspec
bundle exec rubocop
bundle exec brakeman -q --no-summary
```

### The frontend gate (new in this addendum — SEO + accessibility)

This project's product catalog (`/products`) is driven through two additional specialist reviewers,
[`seo-optimizer.md`](.claude/agents/seo-optimizer.md) and
[`frontend-qa.md`](.claude/agents/frontend-qa.md), backed by real tooling. `frontend-qa`'s checks
need a **real browser** — `rack_test` (Capybara's default, no-JS driver) can't run JavaScript, and
axe *is* a JavaScript library that gets injected into the page and evaluated:

```bash
brew install --cask google-chrome     # if you don't already have Chrome
brew install chromedriver             # the WebDriver binary Selenium talks to
```

| Command | What it's for |
|---|---|
| `bundle exec rspec spec/system/accessibility_spec.rb` | axe (`axe-core-rspec`/`axe-core-capybara`, `be_axe_clean`) — zero automated WCAG 2.1 AA violations on the catalog pages. Catches ~30–40% of real accessibility issues; still not a substitute for testing with actual assistive technology. |
| `bundle exec rspec spec/system/seo_spec.rb` | Unique `<title>` per page, the required Open Graph tags, and valid schema.org Product JSON-LD. |
| `bundle exec rake html_proofer:check` | html-proofer — renders the catalog pages to `tmp/html_proofer/` and checks for broken internal links, invalid HTML, and missing `alt` attributes. |

```bash
bundle exec rspec spec/system/accessibility_spec.rb spec/system/seo_spec.rb
bundle exec rake html_proofer:check
```

**Lighthouse CI (`@lhci/cli` 0.15.1, Node) — a reference gate, not required to develop this app:**

```bash
npm install                 # installs @lhci/cli from package.json
bin/rails server -p 3000 &  # lhci's own config (.lighthouserc.json) starts/stops this for you too —
                             # this line is only if you want the server up for manual poking as well
npx lhci autorun             # audits SEO/performance/accessibility/best-practices against .lighthouserc.json
```

`npx lhci autorun` is what it's *for*: a second-opinion, whole-page performance/SEO/accessibility
audit — not a replacement for the RSpec-driven checks above, which is why `.claude/hooks/verify.sh`
references it in its output but does not invoke it (it's a Ruby-only script; Lighthouse CI is a
Node tool with its own toolchain).

## 4. Project layout

```
app/
  controllers/   sessions, registrations, products, carts, line_items, checkout/orders
  models/        User, Session, Current, Product, Cart, LineItem, Order
  services/      PaymentService (the Stripe integration seam, stubbed in test/dev by default)
  helpers/       ProductsHelper (schema.org JSON-LD builders)
  views/         layouts/application.html.erb + one view per controller action
config/
  routes.rb, database.yml, sitemap.rb
db/
  schema.rb (hand-maintained snapshot), seeds.rb
docs/
  architecture.md, definition-of-done.md, features/FEATURE-*.md
spec/
  models/, requests/, system/ — RSpec examples, one file per feature area
.claude/
  agents/   researcher · implementer · reviewer · seo-optimizer · frontend-qa
  hooks/    guard.sh (PreToolUse) · verify.sh (PostToolUse) · context.sh (SessionStart)
public/robots.txt, Gemfile, .rubocop.yml, .env.example, .lighthouserc.json, package.json
```

## 5. Troubleshooting

**"xcrun: error: invalid active developer path" / `bundle install` fails compiling a native gem.**
Xcode Command Line Tools aren't installed (or were removed by a macOS update). Run
`xcode-select --install`, wait for the ~500MB download, then retry.

**`bundle install` fails on `sqlite3` or another gem with a C extension.**
Almost always missing Command Line Tools (above). If it persists, `rbenv exec gem pdk` style
mismatches between your shell's Ruby and rbenv's can be the cause — confirm `which ruby` points
inside `~/.rbenv/versions/4.0.6/...`, not `/usr/bin/ruby` (macOS's system Ruby, which you should
never use for app development).

**Apple Silicon (M-series) and a future Postgres/other compiled dependency.**
Homebrew's prefix is `/opt/homebrew` on Apple Silicon (`/usr/local` on Intel). If a future gem needs
`bundle config build.<gem> --with-<lib>-dir=...`, point it at `/opt/homebrew/opt/<formula>`, not
`/usr/local/opt/<formula>` — this project's SQLite default needs no such step, but it's the #1 snag
if you ever swap in PostgreSQL (`docs/research/NOTE-SDLC-4-ADD-macos-setup.md` §3 has the full
sequence).

**`rbenv: version '4.0.6' is not installed` even after `rbenv install 4.0.6`.**
Your shell hasn't picked up rbenv's shims yet — confirm §1a's `eval "$(rbenv init - zsh)"` line is
actually in `~/.zshrc` (not commented out — a previously-commented rbenv line can make rbenv skip
re-initializing) and open a fresh terminal tab.

**`bundle exec rspec spec/system/*.rb` errors with something like "unable to find chromedriver" or
a WebDriver connection refused.**
`frontend-qa`'s checks need Chrome + a matching `chromedriver` on `PATH` (§3). `bundle exec rspec`
with no path argument (just the model/request specs) does not need a browser at all — run that
first to confirm the rest of the app is healthy before troubleshooting the browser-driven specs.

**`npx lhci autorun` hangs or fails to start the server.**
`.lighthouserc.json`'s `startServerCommand` runs `bin/rails server -p 3000` itself — if port 3000 is
already in use (e.g., you left `bin/rails server` running from §2), stop that process first or edit
the port in both `.lighthouserc.json` and the command you run.

## 6. What's next

Read the book chapter this project ships as a worked example for:
[`04-ai-assisted-sdlc/03-worked-examples/03-rails-estore-sdlc.md`](../../03-rails-estore-sdlc.md) —
it walks the governed loop (spec → ground → implement → gate → review → merge) through this exact
codebase, including the real authorization bug a fresh reviewer caught that RSpec, RuboCop, and
Brakeman all reported clean, and the real accessibility/SEO gaps `frontend-qa`/`seo-optimizer` caught
that neither of those tools was ever built to check.
