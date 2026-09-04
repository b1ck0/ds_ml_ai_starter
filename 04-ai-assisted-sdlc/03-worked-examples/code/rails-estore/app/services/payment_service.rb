# PaymentService — the real Stripe integration point, with the real charge stubbed. This is exactly
# NOTE-SDLC-4-4-stripe-checkout.md's recommended shape: `Stripe::Checkout::Session.create` is the
# real, current, hosted-checkout API call a production deployment would make; in test mode (or with
# STRIPE_STUB=true) it returns a fake session hash and NEVER calls the Stripe client, so this
# project's whole test suite — and this chapter's worked example — runs with no Stripe account and
# no secret key in the repo.
class PaymentService
  def initialize(order)
    @order = order
  end

  def create_checkout_session
    if Rails.env.test? || ENV["STRIPE_STUB"] == "true"
      { id: "cs_test_stub_#{@order.id}", url: "https://checkout.stripe.com/stub" }
    else
      session = Stripe::Checkout::Session.create(
        payment_method_types: ["card"],
        line_items: build_line_items,
        mode: "payment",
        success_url: "#{Rails.application.routes.url_helpers.root_url}checkout/orders/#{@order.id}?session_id={CHECKOUT_SESSION_ID}",
        cancel_url: cancel_url
      )
      { id: session.id, url: session.url }
    end
  end

  private

  def build_line_items
    @order.line_items.map do |item|
      {
        price_data: {
          currency: "usd",
          product_data: { name: item.product.name },
          unit_amount: item.product.price_cents
        },
        quantity: item.quantity
      }
    end
  end

  def cancel_url
    Rails.application.routes.url_helpers.cart_url
  end
end
