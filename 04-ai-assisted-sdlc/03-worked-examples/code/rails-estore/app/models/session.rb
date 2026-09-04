# Session — database-tracked, one row per signed-in browser, holding a signed random token (never
# the raw session id) plus the request metadata the Rails 8 auth generator captures by default.
# Source: NOTE-SDLC-4-2-auth-generator.md ("Sessions table includes unique token, ip_address and
# user_agent fields").
class Session < ApplicationRecord
  belongs_to :user

  has_secure_token :token

  validates :ip_address, presence: true
end
