# LineItem — shared between Cart and Order via a polymorphic `cartable` association, so "checkout"
# is "copy each of the cart's line items onto a new order," not a second schema. Order + LineItem
# association shape: NOTE-SDLC-4-4-stripe-checkout.md.
class LineItem < ApplicationRecord
  belongs_to :product
  belongs_to :cartable, polymorphic: true

  validates :quantity, presence: true, numericality: { only_integer: true, greater_than: 0 }

  def subtotal_cents
    quantity * product.price_cents
  end
end
