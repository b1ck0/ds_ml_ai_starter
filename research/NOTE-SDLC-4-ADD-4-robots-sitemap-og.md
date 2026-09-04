# NOTE-SDLC-4-ADD-4: robots.txt, sitemap.xml Conventions, and Open Graph Meta Tags

**Answer:**
Three SEO/social standards that the e-store must implement:

1. **robots.txt** — a plain-text file at `public/robots.txt` telling crawlers which paths to crawl/skip;
   should reference the sitemap.
2. **sitemap.xml** — a machine-readable XML index of all indexable URLs; Google recommends JSON-LD or
   XML format; max 50,000 URLs / 50 MB per file.
3. **Open Graph meta tags** — four required `<meta>` tags in the document `<head>` for social sharing:
   `og:title`, `og:type`, `og:image`, `og:url`.

All three are standard, well-supported practices with official specs and Google/Facebook guidance.

## Evidence

### robots.txt

**Google Search Central — robots.txt specification:**
- Source: https://developers.google.com/search/docs/crawling-indexing/robots/robots_txt
- Standard format (case-sensitive):
  ```
  User-agent: *
  Disallow: /admin
  Disallow: /private
  Allow: /products
  
  Sitemap: https://example.com/sitemap.xml
  ```
- **User-agent:** Crawler name or `*` (all bots); typically no need to repeat Sitemap for each agent.
- **Disallow:** Path to block (e.g., `/admin` blocks `/admin` and `/admin/anything`; `/` blocks all).
- **Allow:** Exceptions to Disallow (e.g., `Disallow: /temp` followed by `Allow: /temp/public`).
- **Sitemap:** URL of sitemap; include once at top or bottom of file.
- **Best practice:** Do not block important resources (CSS, JS, images) needed for rendering; blocking
  them prevents crawlers from seeing page quality.
- Date checked: 2026-09-04

**Google Search Central — Robots.txt Best Practices:**
- Source: https://developers.google.com/search/docs/crawling-indexing/robots/robots_txt
- Quote: "If a page is disallowed in robots.txt, it should not appear in your XML sitemap, as your
  robots.txt and sitemap must agree."
- Do not list noindex pages, redirects, 404s, or canonical duplicates in your sitemap.
- Date checked: 2026-09-04

### sitemap.xml

**sitemaps.org — XML Sitemap Protocol:**
- Source: https://www.sitemaps.org/ (canonical specification)
- Standard XML format:
  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
      <loc>https://example.com/products/item-1</loc>
      <lastmod>2026-09-04</lastmod>
      <changefreq>weekly</changefreq>
      <priority>0.8</priority>
    </url>
  </urlset>
  ```
- Date checked: 2026-09-04

**Google Search Central — Build and Submit a Sitemap:**
- Source: https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap
- **Required elements:** `<urlset>`, `<url>`, `<loc>` (fully qualified URL)
- **Optional elements:** `<lastmod>` (ISO 8601 date), `<changefreq>` (never, yearly, monthly, weekly,
  daily, hourly), `<priority>` (0.0–1.0, default 0.5)
- **File limits:** Max 50,000 URLs or 50 MB (uncompressed) per sitemap; use sitemap index if larger
  ```xml
  <sitemapindex>
    <sitemap><loc>https://example.com/sitemap-1.xml</loc></sitemap>
    <sitemap><loc>https://example.com/sitemap-2.xml</loc></sitemap>
  </sitemapindex>
  ```
- **Google note:** `<priority>` and `<changefreq>` are largely ignored; focus on accurate `<lastmod>`.
- **Best practice:** Never include noindex pages, 4xx URLs, or canonical variants; only include pages you
  want indexed.
- **Submission:** Register sitemap in Google Search Console and Bing Webmaster Tools, and reference in
  `robots.txt`.
- Date checked: 2026-09-04

**sitemap_generator gem (7.1.1) — Rails integration:**
- Source: https://rubygems.org/gems/sitemap_generator and https://github.com/kjvarga/sitemap_generator
- Generates XML sitemaps dynamically from Rails config DSL; rake task `rake sitemap:refresh` creates
  sitemap.xml and auto-pings search engines.
- Supports video, news, image sitemaps, and image/video tags per URL.
- Can output to local filesystem or remote (S3, etc.).
- Date checked: 2026-09-04

### Open Graph Meta Tags

**Open Graph Protocol (ogp.me):**
- Source: https://ogp.me/
- Four **required** properties:
  ```html
  <meta property="og:title" content="Page Title" />
  <meta property="og:type" content="website" />
  <meta property="og:image" content="https://example.com/image.jpg" />
  <meta property="og:url" content="https://example.com/page" />
  ```
- **Recommended** properties:
  ```html
  <meta property="og:description" content="Brief description" />
  <meta property="og:locale" content="en_US" />
  <meta property="og:site_name" content="My Site" />
  ```
- **og:type:** Determines how content is rendered on social platforms. Common types:
  - `website` (default for pages)
  - `article` (for blog posts)
  - `product` (for product pages)
  - `video.movie`, `music.song`, etc. (for media)
- **og:image:** Must be a fully qualified URL (http/https), JPEG/PNG/GIF, and ideally square with
  dimensions ≥ 1200×1200px (Facebook recommendation: 1200×630px for best display).
- **og:url:** Canonical URL of the page; prevents duplicates.
- Date checked: 2026-09-04

**Facebook / Meta Sharing Debugger:**
- Source: https://developers.facebook.com/tools/debug/ (reference; not directly cited)
- Social platforms use og:title, og:description, og:image, og:url to render a "card" when a link is
  shared.
- Date checked: 2026-09-04

**meta-tags gem integration:**
- Source: https://rubygems.org/gems/meta-tags (2.24.0)
- Simplifies OG tag generation in Rails:
  ```ruby
  # In controller:
  set_meta_tags(
    og: {
      title: "Product Name",
      description: "Description",
      image: "https://example.com/product.jpg",
      type: "product",
      url: "https://example.com/products/123"
    }
  )
  
  # In layout:
  <%= display_meta_tags %>  # Renders all meta tags including og: prefixed
  ```
- Date checked: 2026-09-04

## Canonical URLs

**WCAG / SEO best practice:**
- Quote from Google Search Central: "Include a canonical URL in your sitemaps to avoid confusion."
- Syntax: `<link rel="canonical" href="https://example.com/canonical-url">`
- Prevents indexing of duplicate/parameter variant URLs (e.g., `/products/1?utm_source=email` and
  `/products/1` should both point to the canonical).
- meta-tags gem supports canonical via `set_meta_tags(canonical: "https://...")`
- Date checked: 2026-09-04

## Caveats / Limits

- **robots.txt is advisory, not enforced:** Well-behaved crawlers respect it; bad-faith scrapers ignore it.
  It does not provide security (sensitive data should not be indexed).
- **Sitemap is a hint, not a guarantee:** Google crawls what it chooses; a sitemap only suggests URLs.
  Crawl budget and link popularity still drive actual crawling.
- **Open Graph is platform-specific:** Not all platforms support all og:type values. Facebook, Twitter
  (now X), LinkedIn, etc. may render differently.
- **Image URLs must be absolute:** og:image must start with http:// or https://, not a relative path.
- **Timing:** robots.txt and sitemap changes are picked up by crawlers within hours to days, not
  immediately.
- **Locale variants:** If your site serves multiple languages, use `og:locale` and `og:locale:alternate`
  to signal different language versions.

## Recommendation

- **For the e-store scaffold (FEATURE-3):**
  1. Add `public/robots.txt` with User-agent: *, Disallow: /admin, and Sitemap reference.
  2. Configure `config/sitemap.rb` via sitemap_generator to auto-generate product/category URLs; run
     `rake sitemap:refresh` in deploy/CI.
  3. In the product and category show views, use `set_meta_tags` to set og:title, og:type (product),
     og:image (product image), og:url (canonical URL), og:description.
  4. In the layout, call `display_meta_tags` with `site: "E-Store"` to render all tags.

- **For the SEO-optimizer agent checklist:**
  - Verify robots.txt exists and references sitemap
  - Verify sitemap.xml is accessible and valid (no 404s, correct XML)
  - Verify product pages have og:title, og:image, og:url (required 4)
  - Verify og:image URLs are absolute and resolve
  - Verify canonical URL is present and matches the page's actual URL
  - Verify no duplicate og:title tags across pages (should be unique per page)

- **For the spec:**
  ```ruby
  # spec/system/seo_spec.rb
  describe 'Product page SEO' do
    it 'has correct Open Graph meta tags' do
      visit product_path(@product)
      expect(page.find('meta[property="og:title"]')['content']).to eq(@product.name)
      expect(page.find('meta[property="og:image"]')['content']).to match(/https?:/)
      expect(page.find('meta[property="og:url"]')['content']).to eq(product_url(@product))
    end
  end
  ```
