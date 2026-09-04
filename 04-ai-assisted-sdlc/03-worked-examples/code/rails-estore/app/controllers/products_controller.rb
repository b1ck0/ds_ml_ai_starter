# ProductsController — the catalog is public: browsing does not require sign-in, so this is the one
# controller in the app that deliberately widens ApplicationController's closed-by-default posture,
# and it does so by NAME (`only: %i[index show]`) so nothing else on this controller is accidentally
# exposed if another action is added later.
#
# FEATURE-3 (SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents) adds the `set_meta_tags` calls below — a
# distinct title/description/canonical/Open-Graph block per action, never one hardcoded string reused
# across pages, which is exactly what the seo-optimizer agent's checklist item 1 exists to catch if
# it regresses. `set_meta_tags`/`display_meta_tags`: NOTE-SDLC-4-ADD-1-gem-npm-versions.md.
class ProductsController < ApplicationController
  allow_unauthenticated_access only: %i[index show]

  def index
    @products = if params[:q].present?
                  Product.where("name LIKE ?", "%#{Product.sanitize_sql_like(params[:q])}%")
                else
                  Product.all
                end

    set_meta_tags(
      title: "Shop All Products",
      description: "Browse the full Rails E-Store catalog.",
      canonical: products_url,
      og: {
        title: "Shop All Products",
        type: "website",
        url: products_url,
        image: default_og_image
      }
    )
  end

  def show
    @product = Product.find(params[:id])

    set_meta_tags(
      title: @product.name,
      description: @product.description.presence || "#{@product.name} — Rails E-Store.",
      canonical: product_url(@product),
      og: {
        title: @product.name,
        type: "product",
        url: product_url(@product),
        image: @product.image_url.presence || default_og_image,
        description: @product.description
      }
    )
  end

  private

  # A generic, absolute placeholder so `og:image` is never a relative (and therefore silently
  # broken-on-every-platform) URL even for the catalog index, which has no single product photo of
  # its own. https://placehold.co is a free placeholder-image service; swap for a real brand image.
  def default_og_image
    "https://placehold.co/1200x630?text=Rails+E-Store"
  end
end
