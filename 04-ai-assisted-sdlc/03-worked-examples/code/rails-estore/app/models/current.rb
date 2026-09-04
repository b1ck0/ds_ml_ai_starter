# Current — a per-request singleton (ActiveSupport::CurrentAttributes), populated once per request
# by Authentication#resume_session. Every controller/view reads `Current.user` instead of threading
# the signed-in user through every method call by hand.
# Source: NOTE-SDLC-4-2-auth-generator.md ("Current is a singleton inheriting
# ActiveSupport::CurrentAttributes").
#
# Java/Spring bridge: this plays the role a request-scoped bean (or a ThreadLocal SecurityContext,
# as in Spring Security's `SecurityContextHolder`) plays in a Java web app — a value that's global
# *within* one request's thread, reset before the next.
class Current < ActiveSupport::CurrentAttributes
  attribute :session

  delegate :user, to: :session, allow_nil: true
end
