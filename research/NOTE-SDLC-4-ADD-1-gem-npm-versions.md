# NOTE-SDLC-4-ADD-1: Ruby Gem and npm Package Versions + Invocation

**Answer:**
Five real, current tools verified at source:
- **meta-tags** `2.24.0` — use `set_meta_tags()` in controllers, `display_meta_tags()` in layout `<head>`
- **sitemap_generator** `7.1.1` — run `rake sitemap:refresh` to generate + ping search engines
- **html-proofer** `5.2.2` — invoke `htmlproofer ./out` CLI to validate HTML & check links
- **axe-core-rspec** `4.13.0` — use `expect(page).to be_axe_clean` matcher in system specs
- **axe-core-capybara** `4.13.0` — integrates axe into Capybara WebDriver, use same `be_axe_clean` matcher
- **@lhci/cli** `0.15.1` (npm) — invoke `lhci autorun` to audit SEO/performance/accessibility

## Evidence

**meta-tags gem (2.24.0, released 2026-09-01):**
- Source: https://rubygems.org/gems/meta-tags
- Usage pattern (from https://github.com/kpumuk/meta-tags):
  ```ruby
  # In controller:
  set_meta_tags(title: "Page Title", description: "Page description")
  
  # In layout:
  <%= display_meta_tags site: "My website" %>
  ```
- Requires Ruby >= 3.0.0, 21M+ downloads
- Date checked: 2026-09-04

**sitemap_generator gem (7.1.1, released 2026-07-27):**
- Source: https://rubygems.org/gems/sitemap_generator
- Primary rake task: `rake sitemap:refresh` — generates sitemap.xml + pings Google/Yahoo/Bing
- Config via `config/sitemap.rb` DSL; supports multiple sitemap extensions (video, news, image)
- Date checked: 2026-09-04

**html-proofer gem (5.2.2, released 2026-07-28):**
- Source: https://rubygems.org/gems/html-proofer
- Invocation (from https://github.com/gjtorikian/html-proofer):
  ```bash
  htmlproofer --extensions .html.erb ./out
  ```
- Programmatic usage:
  ```ruby
  HTMLProofer.check_directory("./out").run
  HTMLProofer.check_links(["https://example.com"]).run
  ```
- Checks images (alt tags, broken refs), internal/external links, scripts, favicons, OpenGraph
- Default checks: `['Links', 'Images', 'Scripts']`
- Date checked: 2026-09-04

**axe-core-rspec gem (4.13.0, released 2026-08-11):**
- Source: https://rubygems.org/gems/axe-core-rspec
- RSpec custom matchers for axe accessibility testing
- Usage (from https://github.com/dequelabs/axe-core-gems):
  ```ruby
  expect(page).to be_axe_clean
  expect(page).to be_axe_clean.checking(:label, :tabindex)
  expect(page).to be_axe_clean.according_to(:wcag2a)
  ```
- Requires Ruby >= 2.3.0, 24M+ downloads
- Date checked: 2026-09-04

**axe-core-capybara gem (4.13.0, released 2026-08-11):**
- Source: https://rubygems.org/gems/axe-core-capybara
- Capybara WebDriver with axe injected; works with Capybara system specs (`js: true`)
- Usage (from https://github.com/dequelabs/axe-core-gems):
  ```ruby
  # In spec/system/accessibility_spec.rb
  it 'is accessible', js: true do
    visit '/products'
    expect(page).to be_axe_clean
  end
  ```
- Requires Ruby >= 2.3.0, 4M+ downloads
- Date checked: 2026-09-04

**@lhci/cli npm (0.15.1):**
- Source: https://www.npmjs.com/package/@lhci/cli (from WebSearch results)
- Official repo: https://github.com/GoogleChrome/lighthouse-ci
- Invocation: `lhci autorun` (after building the project in CI/CD)
- Audits: SEO, performance metrics, accessibility, offline support, best practices
- Features: prevents regressions, tracks Lighthouse scores, sets performance budgets
- Date checked: 2026-09-04

## Caveats / Limits

- **Versions are as of 2026-09-04**; check rubygems.org and npmjs.com before production to catch
  security patches or breaking changes.
- **meta-tags, sitemap_generator, html-proofer:** Rails gems; require Rails 6.0+.
- **axe-core-rspec/-capybara:** both gems share the same `be_axe_clean` matcher API and version
  numbering; use both together (rspec matchers + capybara WebDriver integration).
- **@lhci/cli:** Node tool; typical setup requires Node 14+ and a `lighthouserc.json` config file.
- **html-proofer** does **not** catch all HTML validity issues (manual review + W3C Validator recommended
  for exhaustive checks).

## Recommendation

- **For the addendum chapter:** Pin all versions in the Gemfile and package.json; cite these specific
  versions in the tooling section.
- **For the agent checklists:** reference the specific matcher syntax (`be_axe_clean`, `rake
  sitemap:refresh`, `lhci autorun`) so the agents' output can be faithful to actual tool output.
- **For the specs:** Include one full example system spec showing `axe-core-capybara` + `be_axe_clean`
  and one integration test showing html-proofer invocation.
