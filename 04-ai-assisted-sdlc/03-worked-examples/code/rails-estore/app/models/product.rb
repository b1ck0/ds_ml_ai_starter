# Product — the catalog. Price is stored as an integer number of cents (price_cents), never a float
# or a decimal computed in application code: a float dollar amount silently loses precision on
# arithmetic (the classic 0.1 + 0.2 problem), which is unacceptable once money changes hands.
#
# `image_url` and `sku` (FEATURE-3) are deliberately NOT required by a presence validation: FEATURE-2's
# own `spec/requests/checkout_spec.rb` creates `Product.create!(name: "Rails Mug", price_cents: 1_500)`
# with neither, and that file is out of this addendum's scope to touch. The product catalog views
# (app/views/products/show.html.erb) handle a blank image_url by simply omitting the <img> and the
# JSON-LD "image" field rather than assuming it is always present.
class Product < ApplicationRecord
  has_many :line_items, dependent: :restrict_with_error

  validates :name, presence: true
  validates :price_cents, presence: true, numericality: { only_integer: true, greater_than: 0 }
  validates :sku, uniqueness: true, allow_nil: true

  def price_dollars
    price_cents / 100.0
  end

  # schema.org Product / Google Merchant Listing require an ISO-4217 currency code alongside the
  # price (NOTE-SDLC-4-ADD-2-schema-product.md); this storefront only ever sells in USD.
  def price_currency
    "USD"
  end
end
