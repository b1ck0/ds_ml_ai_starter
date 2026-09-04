# LineItemsController — add/remove a product from the SIGNED-IN USER'S OWN cart. `@cart` is always
# resolved through `Current.user`, and `destroy` looks the line item up as a child of that cart
# (`@cart.line_items.find`, not `LineItem.find`) — so a user can never remove an item from someone
# else's cart even by guessing its id.
class LineItemsController < ApplicationController
  before_action :set_cart

  def create
    product = Product.find(params[:product_id])
    line_item = @cart.line_items.find_or_initialize_by(product: product)
    line_item.quantity = line_item.quantity.to_i + quantity_param
    line_item.save!
    redirect_to cart_path, notice: "Added #{product.name} to your cart."
  end

  def destroy
    @cart.line_items.find(params[:id]).destroy
    redirect_to cart_path, notice: "Removed from your cart."
  end

  private

  def set_cart
    @cart = Current.user.cart_or_create
  end

  def quantity_param
    params.fetch(:quantity, 1).to_i.clamp(1, 100)
  end
end
