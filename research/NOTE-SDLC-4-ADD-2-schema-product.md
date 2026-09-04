# NOTE-SDLC-4-ADD-2: schema.org Product JSON-LD Shape + Google Merchant Listing Requirements

**Answer:**
The **schema.org Product** structured data for a JSON-LD `<script type="application/ld+json">` block on an
e-commerce product page requires:
- **Required:** `name`, `image` (URL), `offers` object (with `price`, `priceCurrency`)
- **Strongly recommended:** `description`, `sku`, `brand`, `availability` (on Offer: `InStock`, `OutOfStock`, etc.)
- **Example:** See below. Google Search Central mandates `price > 0` and ISO-4217 currency code for merchant
  listing eligibility (i.e., purchasable product pages).

## Evidence

**schema.org Product type (authoritative schema definition):**
- Source: https://schema.org/Product
- Describes the Product type with all properties; fields marked as part of the "schema" vocabulary.

**Google Search Central — Merchant Listing requirements:**
- Source: https://developers.google.com/search/docs/appearance/structured-data/merchant-listing
- **Required fields (Merchant Listing):**
  - `name` — product title
  - `image` — one or more product photo URLs
  - `offers` — an Offer object containing:
    - `price` — must be > 0 (the current, active offer price)
    - `priceCurrency` — ISO 4217 three-letter code (e.g., "USD", "GBP", "EUR")
- **Recommended fields:**
  - `description` — product details
  - `sku` — seller's stock keeping unit / merchant identifier
  - `brand` — Brand object with `name` field
  - `availability` — (on Offer) e.g. `InStock`, `OutOfStock`, `BackOrder`, `PreOrder`
- Quote: "Merchant listing experiences require a price greater than zero" and "only pages where a shopper
  can purchase a product are eligible for merchant listing experiences."
- Date checked: 2026-09-04

**Google Search Central — Product Snippets (review-focused, lower barrier):**
- Source: https://developers.google.com/search/docs/appearance/structured-data/product-snippet
- Requires: `name` + at least one of: `review`, `aggregateRating`, or `offers`
- For merchant listings (purchasable), use the more complete schema above.
- Date checked: 2026-09-04

## JSON-LD Example

Correct, minimal JSON-LD for a merchant listing product on a product show page:

```json
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Wireless Headphones Model X1",
  "description": "High-quality noise-cancelling wireless headphones with 30-hour battery life.",
  "image": "https://example.com/images/headphones-x1.jpg",
  "sku": "SKU-12345-WH-BLK",
  "brand": {
    "@type": "Brand",
    "name": "AudioTech"
  },
  "offers": {
    "@type": "Offer",
    "price": "199.99",
    "priceCurrency": "USD",
    "availability": "https://schema.org/InStock",
    "url": "https://example.com/products/headphones-x1"
  }
}
```

## Caveats / Limits

- **Required at minimum:** `name`, `image`, `offers.price`, `offers.priceCurrency`. Without these, the page
  is not eligible for merchant listing rich results.
- **availability:** Must use the full schema.org URL (`https://schema.org/InStock`) not a string literal
  ("InStock").
- **Price format:** Must be a numeric string ("199.99"), never "$199.99" or "1,350.00" with thousands
  separators.
- **Multiple images:** `image` can be an array of URLs for different product angles/colors.
- **Variants:** For products with variants (size, color), use `hasVariant` array of Product objects; each
  variant inherits the parent's brand/description unless overridden.
- **Ratings/Reviews:** `aggregateRating` and `review` are optional but recommended for search ranking; they
  can be added to the same Product object.

## Recommendation

- **For the FEATURE-3 product catalog:** Include the full example above in the product show view as a
  `<script type="application/ld+json">` block, using Rails helpers to interpolate `@product.name`,
  `@product.sku`, etc.
- **For the SEO-optimizer agent:** The checklist should verify:
  1. Product JSON-LD is present and valid (not malformed)
  2. `name`, `image`, `offers.price`, `offers.priceCurrency` are all present and non-empty
  3. Price is a number > 0
  4. `priceCurrency` is a valid ISO-4217 code
  5. (Optional) `availability` is one of the schema.org standard values
- **For the spec:** Include a system spec or controller spec that parses the rendered HTML, extracts the
  JSON-LD block, and asserts that required fields are present and valid.
