# config/sitemap.rb — SitemapGenerator DSL (gem sitemap_generator 7.1.1). Run `bin/rails
# sitemap:refresh` (the gem's Railtie exposes it as a Rails task; historically documented as `rake
# sitemap:refresh`) to (re)generate public/sitemap.xml.gz and ping search engines.
# NOTE-SDLC-4-ADD-1-gem-npm-versions.md, NOTE-SDLC-4-ADD-4-robots-sitemap-og.md.
#
# Every URL here must NOT be Disallow'd in public/robots.txt -- Google's own guidance is that the two
# files must agree (NOTE-SDLC-4-ADD-4). /cart and /checkout are Disallow'd there and are correctly
# absent below: they are per-user pages, not indexable catalog content.
SitemapGenerator::Sitemap.default_host = "https://example.com"

SitemapGenerator::Sitemap.create do
  add products_path, changefreq: "daily", priority: 0.8

  Product.find_each do |product|
    add product_path(product), lastmod: product.updated_at, changefreq: "weekly", priority: 0.9
  end
end
