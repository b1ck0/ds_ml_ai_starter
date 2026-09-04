# CartsController — requires sign-in (ApplicationController's default). There is exactly one cart
# per user, looked up (and lazily created) through `Current.user`, never through an id in the URL —
# there is no `params[:id]` to look up here at all, which is the simplest form authorization can take.
class CartsController < ApplicationController
  def show
    @cart = Current.user.cart_or_create
  end
end
