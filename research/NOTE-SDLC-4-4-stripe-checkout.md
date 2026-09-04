# NOTE-SDLC-4-4: Stripe integration and idiomatic checkout/order modelling in Rails 8

**Answer:**

**Idiomatic order modelling:** Order (with status/state field: e.g., `pending → processing → completed → cancelled`) has-many LineItems, each LineItem belongs-to Product and Order. Order also belongs-to User.

**Stripe integration (test mode):** Use the **stripe gem 19.3.0** (June 24, 2026). Modern production flow uses **`Stripe::Checkout::Session`** (simplified vs. older PaymentIntent pattern for hosted checkout). Test mode: credentials (`stripe_publishable_key`, `stripe_secret_key`) come from `Rails.application.credentials` (via `rails credentials:edit`) or `ENV`, never hardcoded. Prefix test keys with `pk_test_` and `sk_test_` from Stripe Dashboard. Stub the actual charge in the payment service to allow test runs without a Stripe account or real secret in the repo.

**Evidence:**

1. **Order + LineItem association pattern (idiomatic Rails)** — Standard Rails e-commerce tutorial pattern, confirmed in [Rails Guides: Association Basics](https://guides.rubyonrails.org/association_basics.html) and widely used in production apps. Order `has_many :line_items` and LineItem `belongs_to :order, :product`.

2. **Order status/state field** — [Rails AASM state machine](https://github.com/aasm/aasm) or simple enum; idiomatic approach in Rails is `enum status: { pending: 0, processing: 1, completed: 2, cancelled: 3 }` on Order model, allowing queries like `Order.where(status: :completed)`.

3. **Stripe gem version 19.3.0** — [All versions of stripe on RubyGems](https://rubygems.org/gems/stripe/versions) shows 19.3.0 released June 24, 2026. [stripe-ruby GitHub releases](https://github.com/stripe/stripe-ruby/releases) confirms this as the latest stable version.

4. **Stripe::Checkout::Session in current production workflow** — [Create a Checkout Session - Stripe API Reference](https://docs.stripe.com/api/checkout/sessions/create?lang=ruby) documents `Stripe::Checkout::Session.create()` as the modern integration point. This replaces older PaymentIntent direct flows for hosted checkout scenarios. [RubyDoc.info: Stripe::Checkout::Session](https://www.rubydoc.info/gems/stripe/Stripe/Checkout/Session) confirms the class exists and is actively documented.

5. **Test mode prefixes (pk_test_, sk_test_)** — [Stripe API versioning and test mode](https://docs.stripe.com/api/versioning?lang=ruby) confirms test keys are prefixed with `test_` and sourced from Stripe Dashboard API Keys section. Test sessions are returned with `cs_test_` identifiers.

6. **Credentials via Rails.application.credentials or ENV** — [Stripe Ruby documentation](https://docs.stripe.com/libraries/ruby) recommends storing API keys in environment variables or Rails credentials. [Rails credentials docs](https://guides.rubyonrails.org/security.html#custom-credentials) state: credentials are encrypted by default and stored in `config/credentials.yml.enc`, decrypted at runtime via the master key.

7. **Stubbing payment service for test mode without account/secret** — Standard pattern in Rails apps; use a `PaymentService` class in `app/services/` that wraps the Stripe call. In test/CI, inject a stub or mock that returns a fake success response without calling Stripe. This allows the chapter's test suite to run without a Stripe account or real key in the repo.

**Caveats / limits:**

- **Stripe::Checkout::Session** is for hosted checkout (redirects user to Stripe-hosted page). If the chapter requires custom payment forms, a PaymentIntent + Elements flow is needed instead; however, hosted checkout is simpler and sufficient for an e-commerce teaching app.
- **Test mode credentials** (`pk_test_`, `sk_test_`) must never be committed. Use `.env.example` (documenting the variable names) and `.env` (local, `.gitignore`d) or Rails credentials for real values. Example `.env.example`:
  ```
  STRIPE_PUBLISHABLE_KEY=pk_test_XXX
  STRIPE_SECRET_KEY=sk_test_XXX
  ```
- **Idempotency:** Order creation should be idempotent (no duplicate orders on retry). Store Stripe session ID in the Order, and check for existing Order with that session ID before creating a new one.
- Stripe gem 19.3.0 requires Ruby ≥ 2.7; compatible with Ruby 4.0.6.
- Date checked: **2026-09-04**.

**Recommendation:**

1. **Model: Order + LineItem**
   ```ruby
   class Order < ApplicationRecord
     belongs_to :user
     has_many :line_items, dependent: :destroy
     has_many :products, through: :line_items
     enum status: { pending: 0, processing: 1, completed: 2, cancelled: 3 }
     
     validates :total_amount, presence: true, numericality: { greater_than: 0 }
   end

   class LineItem < ApplicationRecord
     belongs_to :order
     belongs_to :product
     
     validates :quantity, presence: true, numericality: { greater_than: 0 }
   end
   ```

2. **Credentials in `.env.example`:**
   ```
   STRIPE_PUBLISHABLE_KEY=pk_test_XXX (get from Stripe Dashboard)
   STRIPE_SECRET_KEY=sk_test_XXX (get from Stripe Dashboard)
   ```

3. **PaymentService stub in `app/services/payment_service.rb`:**
   ```ruby
   class PaymentService
     def initialize(order)
       @order = order
     end

     def create_checkout_session
       # In test/CI, stub this to return a mock session without calling Stripe
       if Rails.env.test? || ENV["STRIPE_STUB"] == "true"
         { id: "cs_test_stub_#{@order.id}", url: "https://checkout.stripe.com/stub" }
       else
         Stripe::Checkout::Session.create(
           payment_method_types: ["card"],
           line_items: build_line_items,
           mode: "payment",
           success_url: "#{Rails.application.config.action_mailer.default_url_options[:host]}/checkout/success?session_id={CHECKOUT_SESSION_ID}",
           cancel_url: "#{Rails.application.config.action_mailer.default_url_options[:host]}/checkout/cancel"
         )
       end
     end

     private

     def build_line_items
       @order.line_items.map do |item|
         {
           price_data: {
             currency: "usd",
             product_data: { name: item.product.name },
             unit_amount: (item.product.price * 100).to_i
           },
           quantity: item.quantity
         }
       end
     end
   end
   ```

4. **Wiring in checkout controller:** After Order is created, call `PaymentService.new(order).create_checkout_session` to get the session. Redirect to `session.url` (or in test mode, directly mark order as completed).

This approach allows the chapter's tests and local runs to pass without Stripe credentials while demonstrating the real integration point for production.
