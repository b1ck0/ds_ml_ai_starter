require "rails_helper"
require "json"

# FEATURE-3 (SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents) — unique <title> per page, the four
# required Open Graph properties, and valid schema.org Product JSON-LD carrying the Google Merchant
# Listing required fields (name, image, offers.price > 0, offers.priceCurrency).
# NOTE-SDLC-4-ADD-2-schema-product.md, NOTE-SDLC-4-ADD-4-robots-sitemap-og.md. See this chapter's
# FEATURE-3 loop transcript for the version of products/show.html.erb that shipped with NO JSON-LD
# block at all (the seo-optimizer finding) and the fix that made the second example below pass.
RSpec.describe "Product catalog SEO", type: :system do
  let(:mug) do
    Product.create!(
      name: "Rails Mug",
      description: "A mug for Rubyists.",
      price_cents: 1_500,
      sku: "MUG-001",
      image_url: "https://placehold.co/600x600?text=Rails+Mug"
    )
  end

  it "renders a unique, descriptive <title> per page" do
    visit products_path
    index_title = page.title

    visit product_path(mug)
    show_title = page.title

    expect(index_title).not_to eq(show_title)
    expect(show_title).to include("Rails Mug")
  end

  it "renders valid Product JSON-LD with the required merchant-listing fields" do
    visit product_path(mug)

    script = find('script[type="application/ld+json"]', match: :first, visible: false)
    json_ld = JSON.parse(script.text(:all))

    expect(json_ld["@type"]).to eq("Product")
    expect(json_ld["name"]).to eq("Rails Mug")
    expect(json_ld["image"]).to match(%r{\Ahttps?://})
    expect(json_ld["offers"]["price"].to_f).to be > 0
    expect(json_ld["offers"]["priceCurrency"]).to eq("USD")
    expect(json_ld["offers"]["availability"]).to eq("https://schema.org/InStock")
  end

  it "renders the four required Open Graph tags with an absolute og:image URL" do
    visit product_path(mug)

    expect(page).to have_css('meta[property="og:title"]', visible: false)
    expect(find('meta[property="og:type"]', visible: false)["content"]).to eq("product")
    expect(find('meta[property="og:image"]', visible: false)["content"]).to match(%r{\Ahttps?://})
    expect(find('meta[property="og:url"]', visible: false)["content"]).to match(%r{/products/#{mug.id}\z})
  end
end
