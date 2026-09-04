# Checkout::OrdersController — cart -> Order, with the PaymentService seam. `show` is the
# authorization-sensitive action (FEATURE-2 AC5): it MUST resolve through `Current.user.orders`, not
# a bare `Order.find(params[:id])`, or any signed-in user can read any other user's order by
# incrementing the id in the URL — an IDOR (insecure direct object reference), OWASP's name for
# exactly this bug. See docs/features/FEATURE-2-checkout.md and this chapter's checkout loop
# transcript for the reviewer catching this the first time it was written wrong.
module Checkout
  class OrdersController < ApplicationController
    before_action :set_cart, only: %i[new create]

    def new
    end

    def create
      if @cart.line_items.none?
        redirect_to cart_path, alert: "Количката ви е празна." and return
      end

      order = build_order_from_cart
      session_data = PaymentService.new(order).create_checkout_session
      order.update!(stripe_session_id: session_data[:id], status: :processing)
      @cart.line_items.destroy_all

      redirect_to checkout_order_path(order), notice: "Поръчката е направена."
    end

    def show
      # Correct, authorized lookup — scoped to the signed-in user, never a bare Order.find.
      @order = Current.user.orders.find(params[:id])
    end

    private

    def set_cart
      @cart = Current.user.cart_or_create
    end

    def build_order_from_cart
      order = nil
      ActiveRecord::Base.transaction do
        order = Current.user.orders.create!(status: :pending)
        @cart.line_items.each do |item|
          order.line_items.create!(product: item.product, quantity: item.quantity)
        end
      end
      order
    end
  end
end
