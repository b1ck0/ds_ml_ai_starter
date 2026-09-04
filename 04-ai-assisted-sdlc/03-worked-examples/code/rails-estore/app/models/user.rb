# User — Rails 8 native authentication generator shape: has_secure_password bcrypt-hashes the
# password into `password_digest`; the plaintext is never stored, logged, or compared directly.
# Source: NOTE-SDLC-4-2-auth-generator.md, NOTE-SDLC-4-5-auth-security.md.
class User < ApplicationRecord
  has_secure_password

  has_many :sessions, dependent: :destroy
  has_one :cart, dependent: :destroy
  has_many :orders, dependent: :destroy

  normalizes :email_address, with: ->(email) { email.strip.downcase }

  validates :email_address, presence: true, uniqueness: true,
                             format: { with: URI::MailTo::EMAIL_REGEXP }
  validates :password, length: { minimum: 8 }, allow_nil: true

  # `admin` is deliberately NOT settable through has_secure_password / mass assignment anywhere in
  # this codebase — see RegistrationsController#user_params. It exists on the schema for the
  # security worked example: FEATURE-1's spec requires a test proving it can't be set this way.

  # Every cart-touching controller (CartsController, LineItemsController, Checkout::OrdersController)
  # needs "this user's cart, creating it on first use" — one method here instead of three copies of
  # `Current.user.cart || Current.user.create_cart!` scattered across controllers.
  def cart_or_create
    cart || create_cart!
  end
end
