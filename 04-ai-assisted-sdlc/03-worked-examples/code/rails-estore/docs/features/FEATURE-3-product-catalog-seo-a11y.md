# FEATURE-3: Product catalog — SEO-complete, accessible storefront pages

**Status:** approved
**Owner:** architect (Opus)
**Routing:** implementer=Sonnet · research=NOTE-SDLC-4-ADD-1, NOTE-SDLC-4-ADD-2, NOTE-SDLC-4-ADD-3,
NOTE-SDLC-4-ADD-4 · review=Sonnet (fresh) + `seo-optimizer` (Sonnet) + `frontend-qa` (Sonnet)

## Intent

FEATURE-1 and FEATURE-2 gave this store an account system and a checkout — but the actual product
pages a shopper (or a search engine, or a share link on social media) meets first were never SEO- or
accessibility-complete: no `<title>`/description per page, no structured data, no `robots.txt`/
sitemap, and — until this feature — no `app/views/layouts/application.html.erb` at all, so there was
nowhere for `<html lang>`, a viewport meta tag, or a skip link to live. This feature closes that gap
and is the vehicle for two new specialist reviewers this addendum adds to the roster:
**`seo-optimizer`** (discoverability) and **`frontend-qa`** (accessibility/frontend quality) — neither
overlaps `reviewer.md`'s authorization/mass-assignment focus, and neither overlaps RuboCop/Brakeman,
which have no opinion on a missing `<label>` or a missing Product JSON-LD block.

## Acceptance criteria

- AC1 — `GET /products` (index) renders exactly one `<h1>`, a search form whose text input has a
  programmatically associated `<label>` (not placeholder-only), and — for every product with an
  `image_url` — an `<img>` with descriptive `alt` text.
- AC2 — `GET /products/:id` (show) renders exactly one `<h1>` and, when the product has an
  `image_url`, an `<img>` with descriptive `alt` text.
- AC3 — every page (`index`, `show`) sets a unique `<title>` and meta description via
  `set_meta_tags`/`display_meta_tags` (the `meta-tags` gem) — `index`'s title and `show`'s title are
  never identical — and a `<link rel="canonical">` pointing at that page's own URL
  [source: NOTE-SDLC-4-ADD-1-gem-npm-versions.md, NOTE-SDLC-4-ADD-4-robots-sitemap-og.md].
- AC4 — every page carries the four required Open Graph properties (`og:title`, `og:type`,
  `og:image`, `og:url`) with an absolute `og:image` URL [source: NOTE-SDLC-4-ADD-4-robots-sitemap-og.md].
- AC5 — the product show page carries valid schema.org **Product** JSON-LD with, at minimum, `name`,
  `image`, `offers.price` (a numeric string > 0), `offers.priceCurrency` ("USD"); `sku`, `brand`, and
  `offers.availability` (`https://schema.org/InStock`) present as the recommended fields, plus a
  **BreadcrumbList** JSON-LD naming Home → Products → the product
  [source: NOTE-SDLC-4-ADD-2-schema-product.md].
- AC6 — `public/robots.txt` exists, references the sitemap, and does not `Disallow` `/products`;
  `config/sitemap.rb` (`sitemap_generator`) lists the products index and every product's show URL
  [source: NOTE-SDLC-4-ADD-4-robots-sitemap-og.md].
- AC7 — `spec/system/accessibility_spec.rb`'s `expect(page).to be_axe_clean.according_to(:wcag21aa)`
  passes on both the index and show pages — zero automated WCAG 2.1 AA violations
  [source: NOTE-SDLC-4-ADD-1-gem-npm-versions.md, NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md].
- AC8 — `bundle exec rake html_proofer:check` reports no broken internal links and no missing `alt`
  attributes on the rendered index/show HTML [source: NOTE-SDLC-4-ADD-1-gem-npm-versions.md].
- AC9 — `app/views/layouts/application.html.erb` sets `<html lang="en">`, includes a skip-to-content
  link that becomes visible on keyboard focus, and structures the page with `<header>`/`<nav>`/
  `<main>`/`<footer>` landmarks [source: NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md].

## Claims to ground

- `meta-tags`/`sitemap_generator`/`html-proofer`/`axe-core-rspec`/`axe-core-capybara`/`@lhci/cli`
  versions and invocation — grounded, `docs/research/NOTE-SDLC-4-ADD-1-gem-npm-versions.md`.
- schema.org Product JSON-LD shape + Google Merchant Listing required/recommended fields — grounded,
  `docs/research/NOTE-SDLC-4-ADD-2-schema-product.md`.
- WCAG/axe checks (labels, alt, contrast, heading order, lang, focus, landmarks) — grounded,
  `docs/research/NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md`.
- `robots.txt`/`sitemap.xml` conventions and Open Graph required properties — grounded,
  `docs/research/NOTE-SDLC-4-ADD-4-robots-sitemap-og.md`.

## Out of scope

- Category/collection pages and the deeper breadcrumb trail they would need (this catalog is a
  single flat `/products` index; the BreadcrumbList AC5 asks for is always exactly three levels deep).
- `AggregateRating`/`Review` schema — no reviews feature exists yet to source them from.
- A themed colour system — this feature relies on the browser's default black-on-white body text,
  which already clears the 4.5:1 contrast minimum; a design-system pass with custom colours is a
  separate feature that would need its own contrast audit.
- Actually running `npx lhci autorun` in this project's own CI — it is wired as a documented,
  grounded **reference** gate (`.lighthouserc.json`, `package.json`) a reader reproduces on a machine
  with Node, per the addendum's scope; `.claude/hooks/verify.sh` does not invoke it.

## Assets to produce

- `.claude/agents/seo-optimizer.md`, `.claude/agents/frontend-qa.md`
- `app/controllers/products_controller.rb` (extended: `set_meta_tags`, search)
- `app/views/products/index.html.erb`, `app/views/products/show.html.erb` (extended)
- `app/views/layouts/application.html.erb` (new)
- `app/helpers/products_helper.rb` (new — `product_json_ld`, `breadcrumb_json_ld`)
- `app/models/product.rb` (extended: `image_url`, `sku`, `price_currency`)
- `db/schema.rb` (extended: `products.image_url`, `products.sku`)
- `db/seeds.rb`, `config/database.yml` (new)
- `public/robots.txt`, `config/sitemap.rb` (new)
- `spec/system/accessibility_spec.rb`, `spec/system/seo_spec.rb`, `spec/support/capybara_driver.rb`
  (new)
- `lib/tasks/html_proofer.rake` (new)
- `.lighthouserc.json`, `package.json` (new — the Node reference gate's config)
- `Gemfile`, `.claude/hooks/verify.sh`, `CLAUDE.md`, `docs/definition-of-done.md` (wiring)

## Gates

Entry: this spec approved; FEATURE-1 + FEATURE-2 merged (the catalog pages reuse `Product` and
`authenticated?` from those features); the four `NOTE-SDLC-4-ADD-*` grounding notes landed. Exit:
`docs/definition-of-done.md` checklist in full, including the new **SEO & Accessibility** section —
with sign-off from `seo-optimizer.md` AND `frontend-qa.md`, in addition to the general
`reviewer.md`'s sign-off.
