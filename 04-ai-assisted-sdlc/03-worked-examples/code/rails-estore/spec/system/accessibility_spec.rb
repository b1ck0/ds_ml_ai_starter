require "rails_helper"
require_relative "../support/capybara_driver"
require "axe-rspec"

# FEATURE-3 (SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents) — axe-core-capybara + axe-core-rspec's
# `be_axe_clean` matcher (both 4.13.0) against the two catalog pages. See this chapter's FEATURE-3
# loop transcript for the version of products/index.html.erb that FAILED this example (a search
# <input> with no associated <label>) and the fix that made it pass.
# NOTE-SDLC-4-ADD-1-gem-npm-versions.md, NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md.
RSpec.describe "Product catalog accessibility", type: :system do
  let(:mug) do
    Product.create!(
      name: "Rails Mug",
      description: "A mug for Rubyists.",
      price_cents: 1_500,
      sku: "MUG-001",
      image_url: "https://placehold.co/600x600?text=Rails+Mug"
    )
  end

  it "has no automatically detectable WCAG 2.1 AA violations on the catalog index", js: true do
    mug
    visit products_path

    expect(page).to be_axe_clean.according_to(:wcag21aa)
  end

  it "has no automatically detectable WCAG 2.1 AA violations on a product page", js: true do
    visit product_path(mug)

    expect(page).to be_axe_clean.according_to(:wcag21aa)
  end
end
