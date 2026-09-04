# Cart — one per user, holding the in-progress selection before checkout. Line items live on the
# polymorphic `LineItem` (see app/models/line_item.rb); checkout copies them onto a new Order and
# then empties this cart, rather than "promoting" the same rows (Order and Cart have independent
# lifecycles — an order must survive the cart being emptied for the next shopping trip).
class Cart < ApplicationRecord
  belongs_to :user
  has_many :line_items, as: :cartable, dependent: :destroy
  has_many :products, through: :line_items

  def total_cents
    line_items.sum(&:subtotal_cents)
  end
end
