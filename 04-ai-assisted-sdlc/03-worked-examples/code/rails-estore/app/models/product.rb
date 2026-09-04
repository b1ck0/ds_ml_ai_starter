# Product — the catalog. Price is stored as an integer number of cents (price_cents), never a float
# or a decimal computed in application code: a float dollar amount silently loses precision on
# arithmetic (the classic 0.1 + 0.2 problem), which is unacceptable once money changes hands.
class Product < ApplicationRecord
  has_many :line_items, dependent: :restrict_with_error

  validates :name, presence: true
  validates :price_cents, presence: true, numericality: { only_integer: true, greater_than: 0 }

  def price_dollars
    price_cents / 100.0
  end
end
