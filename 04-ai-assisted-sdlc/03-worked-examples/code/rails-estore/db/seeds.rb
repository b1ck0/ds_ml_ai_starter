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
    name: "Чаша Rails",
    description: "Керамична чаша за Ruby разработчици. Побира 350 мл кафе и нула N+1 заявки.",
    price_cents: 1_500,
    image_url: "https://placehold.co/600x600?text=Rails+Mug"
  },
  {
    sku: "TSHIRT-001",
    name: "Тениска Convention Over Configuration",
    description: "100% памук. Предлага се в точно един размер — правилния.",
    price_cents: 2_500,
    image_url: "https://placehold.co/600x600?text=Rails+T-Shirt"
  },
  {
    sku: "STICKER-001",
    name: "Пакет стикери Omakase",
    description: "Пет винилови стикера. Устойчиви на атмосферни влияния, устойчиви на капака на лаптопа.",
    price_cents: 800,
    image_url: "https://placehold.co/600x600?text=Sticker+Pack"
  },
  {
    sku: "NOTEBOOK-001",
    name: "Тефтер за миграции",
    description: "192 страници на точки. За разлика от истинска миграция, никога не е обратим.",
    price_cents: 1_200,
    image_url: "https://placehold.co/600x600?text=Notebook"
  }
].each do |attrs|
  Product.find_or_create_by!(sku: attrs[:sku]) do |product|
    product.assign_attributes(attrs)
  end
end

puts "Seeded #{Product.count} product(s)."
