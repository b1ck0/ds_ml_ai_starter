# ApplicationController — every controller requires authentication by default (Authentication's
# `before_action :require_authentication`). A controller opts OUT explicitly and by name
# (`allow_unauthenticated_access only: %i[index show]`) rather than opting in — "closed by default"
# is the same posture a Spring Security `FilterSecurityInterceptor` configured with
# `.anyRequest().authenticated()` takes: every new endpoint is protected unless someone deliberately
# widens it, so a forgotten `before_action` can never silently leave a route open.
class ApplicationController < ActionController::Base
  include Authentication

  allow_browser versions: :modern
end
