# Authentication — the Rails 8 native auth generator's concern, included into ApplicationController
# so every controller requires a signed-in user unless it opts out with `allow_unauthenticated_access`.
# Source: NOTE-SDLC-4-2-auth-generator.md.
#
# Two different "sessions" appear below and they are NOT the same thing — worth being explicit about
# for a reader coming from Java, where "the session" usually means one thing (HttpSession):
#   * `session` (lowercase) is Rails' own encrypted, server-signed cookie jar — used here only to
#     stash the URL the user was headed to before being bounced to sign in.
#   * `Session` (capitalized, app/models/session.rb) is OUR ActiveRecord model — one database row
#     per signed-in browser, looked up by a signed token cookie. It is what `Current.user` resolves
#     through, and what a "log out everywhere" feature would delete all of for a given user.
module Authentication
  extend ActiveSupport::Concern

  included do
    before_action :require_authentication
    helper_method :authenticated?
  end

  class_methods do
    def allow_unauthenticated_access(**options)
      skip_before_action :require_authentication, **options
    end
  end

  private

  def authenticated?
    resume_session.present?
  end

  def require_authentication
    resume_session || request_authentication
  end

  def resume_session
    Current.session ||= find_session_by_cookie
  end

  def find_session_by_cookie
    return nil unless (token = cookies.signed[:session_token])

    Session.find_by(token: token)
  end

  def request_authentication
    session[:return_to_after_authenticating] = request.url
    redirect_to new_session_path
  end

  def after_authentication_url
    session.delete(:return_to_after_authenticating) || root_path
  end

  def start_new_session_for(user)
    user.sessions.create!(user_agent: request.user_agent, ip_address: request.remote_ip).tap do |new_session|
      Current.session = new_session
      cookies.signed.permanent[:session_token] = { value: new_session.token, httponly: true, same_site: :lax }
    end
  end

  def terminate_session
    Current.session&.destroy
    cookies.delete(:session_token)
  end
end
