require "rails_helper"

# Covers FEATURE-2 AC3 (status is a real enum) and the LineItem/total_cents arithmetic AC1 depends on.
RSpec.describe Order, type: :model do
  let(:user) { User.create!(email_address: "buyer@example.com", password: "password1", password_confirmation: "password1") }
  let(:product) { Product.create!(name: "Rails Mug", price_cents: 1_500) }

  it "exposes status as a named enum, not a free-text column (FEATURE-2 AC3)" do
    order = Order.create!(user: user, status: :pending)

    expect(Order.statuses).to eq("pending" => 0, "processing" => 1, "completed" => 2, "cancelled" => 3)
    expect(order.pending?).to be true

    order.processing!
    expect(order.status).to eq("processing")
  end

  it "sums line-item subtotals for total_cents" do
    order = Order.create!(user: user, status: :pending)
    order.line_items.create!(product: product, quantity: 3)

    expect(order.total_cents).to eq(4_500)
  end
end
