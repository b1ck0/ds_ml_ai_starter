# ProductsController — the catalog is public: browsing does not require sign-in, so this is the one
# controller in the app that deliberately widens ApplicationController's closed-by-default posture,
# and it does so by NAME (`only: %i[index show]`) so nothing else on this controller is accidentally
# exposed if another action is added later.
class ProductsController < ApplicationController
  allow_unauthenticated_access only: %i[index show]

  def index
    @products = Product.all
  end

  def show
    @product = Product.find(params[:id])
  end
end
