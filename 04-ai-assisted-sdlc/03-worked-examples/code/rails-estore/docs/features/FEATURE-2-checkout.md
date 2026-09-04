# FEATURE-2: Cart to order — checkout with a stubbed payment seam

**Status:** approved
**Owner:** architect (Opus)
**Routing:** implementer=Sonnet · research=NOTE-SDLC-4-1, NOTE-SDLC-4-4 · review=Sonnet (fresh)

## Intent

A signed-in user (FEATURE-1) can add products to a cart and check out: the cart's line items become
an `Order`'s line items, the order moves through a status enum
(`pending → processing → completed/cancelled`), and a `PaymentService` seam calls out to
`Stripe::Checkout::Session` in test mode — real integration point, stubbed charge, no live key, no
Stripe account required to run the suite [source: NOTE-SDLC-4-4-stripe-checkout.md]. Every order
lookup is scoped to the signed-in user: this is the feature where a missing authorization check does
the most damage — an unscoped `Order.find(params[:id])` lets any signed-in user read (and, if the
action allowed it, act on) anyone else's order by guessing a small integer.

## Acceptance criteria

- AC1 — `POST /checkout/orders` with a non-empty cart creates one `Order` (`status: pending`, then
  `processing` once `PaymentService` returns a session), one `LineItem` per distinct product in the
  cart (copied, not moved, until the transaction commits), and empties the cart.
- AC2 — `POST /checkout/orders` with an EMPTY cart creates no `Order` and redirects back to the cart
  with an alert.
- AC3 — `Order#status` is one of `pending`, `processing`, `completed`, `cancelled` (a Rails `enum`,
  not a free-text column) [source: NOTE-SDLC-4-4-stripe-checkout.md].
- AC4 — `PaymentService#create_checkout_session`, when `Rails.env.test?` (or `STRIPE_STUB=true`),
  returns a stub session hash and makes NO network call — verified by stubbing/mocking
  `Stripe::Checkout::Session.create` and asserting it is never invoked during the test-mode path
  [source: NOTE-SDLC-4-4-stripe-checkout.md].
- AC5 — `GET /checkout/orders/:id` returns the order when it belongs to the signed-in user, and
  returns 404 (via `Current.user.orders.find`, which raises `ActiveRecord::RecordNotFound` for an id
  outside that user's own orders — not a bare `Order.find`) when it belongs to a DIFFERENT user. This
  is the authorization case; it is not satisfied by "the route requires sign-in" alone.
- AC6 — repeating `POST /checkout/orders` for the same cart after a first successful checkout does
  not silently double-charge: the cart is empty after AC1, so a second submission hits AC2's
  empty-cart path.

## Claims to ground

- Order/LineItem association shape, `enum status:` idiom — grounded,
  `docs/research/NOTE-SDLC-4-4-stripe-checkout.md`.
- Stripe gem version, `Stripe::Checkout::Session` as the current hosted-checkout integration point,
  test-mode key prefixes, credentials sourcing — grounded,
  `docs/research/NOTE-SDLC-4-4-stripe-checkout.md`.
- Brakeman's mass-assignment/SQLi/XSS check categories, referenced when reviewing this feature's
  controller — grounded, `docs/research/NOTE-SDLC-4-3-brakeman-checks.md`.

## Out of scope

- A real Stripe webhook handling an async payment confirmation — the seam is stubbed; wiring a real
  webhook is a follow-on feature once a Stripe account exists.
- Refunds, partial cancellation, inventory/stock decrement — not needed to demonstrate the governed
  loop; each would be its own feature spec.
- A custom Stripe Elements payment form — hosted Checkout (`Stripe::Checkout::Session`) is simpler
  and sufficient here [source: NOTE-SDLC-4-4-stripe-checkout.md, "Caveats"].

## Assets to produce

- `app/models/product.rb`, `app/models/cart.rb`, `app/models/line_item.rb`, `app/models/order.rb`
- `app/controllers/products_controller.rb`, `app/controllers/carts_controller.rb`,
  `app/controllers/line_items_controller.rb`, `app/controllers/checkout/orders_controller.rb`
- `app/services/payment_service.rb`
- `app/views/products/*`, `app/views/carts/show.html.erb`, `app/views/checkout/orders/*`
- `db/schema.rb` additions (products, carts, line_items, orders tables)
- `spec/models/order_spec.rb`, `spec/requests/checkout_spec.rb`

## Gates

Entry: this spec approved; FEATURE-1 merged (checkout requires a signed-in user); the two grounding
notes above landed. Exit: `docs/definition-of-done.md` checklist, in full — including AC5's
authorization case with an explicit RSpec example (a second user attempting to read the first user's
order).
