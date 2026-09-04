# Stripe API key — sourced from Rails encrypted credentials first, then ENV, NEVER hardcoded.
# Source: NOTE-SDLC-4-4-stripe-checkout.md ("Credentials via Rails.application.credentials or ENV").
# In test/dev this key is never actually used: PaymentService short-circuits to a stub whenever
# Rails.env.test? or STRIPE_STUB=true (see app/services/payment_service.rb), so a missing or fake
# key here does not break anything.
Stripe.api_key =
  Rails.application.credentials.dig(:stripe, :secret_key) || ENV["STRIPE_SECRET_KEY"]
