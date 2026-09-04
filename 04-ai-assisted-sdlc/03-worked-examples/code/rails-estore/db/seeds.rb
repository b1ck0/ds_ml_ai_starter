# db/seeds.rb — a handful of sample products so `bin/rails db:setup`/`db:seed` leaves the catalog
# non-empty on a friend's first run (README.md). `find_or_create_by!` on `sku` makes this safe to
# re-run (`bin/rails db:seed` again does not create duplicates). Image URLs are absolute
# https://placehold.co placeholders (a free placeholder-image service) — required for the Product
# JSON-LD's "image" field and the Open Graph "og:image" tag to both be valid absolute URLs
# (NOTE-SDLC-4-ADD-2-schema-product.md, NOTE-SDLC-4-ADD-4-robots-sitemap-og.md); swap for real product
# photography before going live.
[
  {
    sku: "MUG-001",
    name: "Rails Mug",
    description: "A ceramic mug for Rubyists. Holds 350ml of coffee and zero N+1 queries.",
    price_cents: 1_500,
    image_url: "https://placehold.co/600x600?text=Rails+Mug"
  },
  {
    sku: "TSHIRT-001",
    name: "Convention Over Configuration T-Shirt",
    description: "100% cotton. Comes in exactly one size: correct.",
    price_cents: 2_500,
    image_url: "https://placehold.co/600x600?text=Rails+T-Shirt"
  },
  {
    sku: "STICKER-001",
    name: "Omakase Sticker Pack",
    description: "Five vinyl stickers. Weatherproof, laptop-lid-proof.",
    price_cents: 800,
    image_url: "https://placehold.co/600x600?text=Sticker+Pack"
  },
  {
    sku: "NOTEBOOK-001",
    name: "Migration Notebook",
    description: "192 dot-grid pages. Never actually reversible, unlike a real migration.",
    price_cents: 1_200,
    image_url: "https://placehold.co/600x600?text=Notebook"
  }
].each do |attrs|
  Product.find_or_create_by!(sku: attrs[:sku]) do |product|
    product.assign_attributes(attrs)
  end
end

puts "Seeded #{Product.count} product(s)."
