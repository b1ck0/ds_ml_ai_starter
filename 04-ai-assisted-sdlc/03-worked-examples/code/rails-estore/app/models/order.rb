# Order — belongs to the user who placed it; status is a Rails `enum` (an int column with named
# values), not a free-text string, so an invalid status is a schema-level impossibility, not a typo
# waiting to happen. Source: NOTE-SDLC-4-4-stripe-checkout.md.
class Order < ApplicationRecord
  belongs_to :user
  has_many :line_items, as: :cartable, dependent: :destroy
  has_many :products, through: :line_items

  enum :status, { pending: 0, processing: 1, completed: 2, cancelled: 3 }

  validates :stripe_session_id, uniqueness: true, allow_nil: true

  def total_cents
    line_items.sum(&:subtotal_cents)
  end
end
