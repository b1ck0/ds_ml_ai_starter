require "rails_helper"

# Covers FEATURE-2 AC1, AC2, AC4, AC5. AC5 is the authorization security case — see this chapter's
# checkout loop transcript for the version of Checkout::OrdersController#show that FAILED this
# example (Order.find without scoping to Current.user) and the fix that made it pass.
RSpec.describe "Checkout", type: :request do
  def sign_in_as(user)
    post session_path, params: { email_address: user.email_address, password: "password1" }
  end

  let(:mug) { Product.create!(name: "Rails Mug", price_cents: 1_500) }
  let(:buyer) do
    User.create!(email_address: "buyer@example.com", password: "password1", password_confirmation: "password1")
  end
  let(:other_user) do
    User.create!(email_address: "other@example.com", password: "password1", password_confirmation: "password1")
  end

  describe "POST /checkout/orders" do
    it "turns a non-empty cart into an order and empties the cart (AC1)" do
      sign_in_as(buyer)
      cart = buyer.create_cart!
      cart.line_items.create!(product: mug, quantity: 2)

      expect {
        post checkout_orders_path
      }.to change { buyer.orders.count }.by(1)

      order = buyer.orders.last
      expect(order.status).to eq("processing")
      expect(order.line_items.count).to eq(1)
      expect(order.line_items.first.quantity).to eq(2)
      expect(cart.reload.line_items).to be_empty
    end

    it "does not call the real Stripe client in test mode (AC4)" do
      sign_in_as(buyer)
      cart = buyer.create_cart!
      cart.line_items.create!(product: mug, quantity: 1)

      expect(Stripe::Checkout::Session).not_to receive(:create)

      post checkout_orders_path
    end

    it "refuses to create an order for an empty cart (AC2)" do
      sign_in_as(buyer)
      buyer.create_cart!

      expect {
        post checkout_orders_path
      }.not_to change { Order.count }

      expect(response).to redirect_to(cart_path)
    end
  end

  describe "GET /checkout/orders/:id — authorization (AC5)" do
    it "returns the order to its own owner" do
      sign_in_as(buyer)
      order = buyer.orders.create!(status: :completed)

      get checkout_order_path(order)

      expect(response).to have_http_status(:ok)
    end

    it "does NOT return another user's order — the IDOR case" do
      others_order = other_user.orders.create!(status: :completed)

      sign_in_as(buyer)

      expect {
        get checkout_order_path(others_order)
      }.to raise_error(ActiveRecord::RecordNotFound)
    end
  end
end
