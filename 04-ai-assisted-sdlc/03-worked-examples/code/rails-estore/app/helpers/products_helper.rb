# ProductsHelper — builds the schema.org structured-data hashes for a product show page as plain
# Ruby Hashes (not inline in the view), so spec/system/seo_spec.rb and the seo-optimizer agent's
# checklist have one obvious place to check for the required/recommended fields, rather than parsing
# them back out of ERB.
#
# Required Google Merchant Listing fields: name, image, offers.price (> 0), offers.priceCurrency.
# Recommended: description, sku, brand, offers.availability (the FULL schema.org URL form, not the
# bare string). Source: NOTE-SDLC-4-ADD-2-schema-product.md (Google Search Central + schema.org).
module ProductsHelper
  def product_json_ld(product)
    {
      "@context" => "https://schema.org",
      "@type" => "Product",
      "name" => product.name,
      "description" => product.description,
      "image" => product.image_url,
      "sku" => product.sku,
      "brand" => { "@type" => "Brand", "name" => "Rails E-Store" },
      "offers" => {
        "@type" => "Offer",
        "price" => format("%.2f", product.price_dollars),
        "priceCurrency" => product.price_currency,
        "availability" => "https://schema.org/InStock",
        "url" => product_url(product)
      }
    }.compact
  end

  # A minimal BreadcrumbList: Home > Products > <this product> — the seo-optimizer checklist's
  # item 5. This catalog is a single flat list with no category pages, so the trail is always
  # exactly three deep; a category feature would extend this, not replace it.
  def breadcrumb_json_ld(product)
    {
      "@context" => "https://schema.org",
      "@type" => "BreadcrumbList",
      "itemListElement" => [
        { "@type" => "ListItem", "position" => 1, "name" => "Home", "item" => root_url },
        { "@type" => "ListItem", "position" => 2, "name" => "Products", "item" => products_url },
        { "@type" => "ListItem", "position" => 3, "name" => product.name, "item" => product_url(product) }
      ]
    }
  end
end
