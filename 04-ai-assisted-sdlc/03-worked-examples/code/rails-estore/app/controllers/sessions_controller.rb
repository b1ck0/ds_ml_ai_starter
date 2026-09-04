# SessionsController — sign in / sign out. Deliberately uses `User.find_by(...).authenticate(...)`
# rather than a single combined lookup, per NOTE-SDLC-4-5-auth-security.md's grounded description of
# `has_secure_password`'s `authenticate(plaintext)` method — the exact API this project's grounding
# note confirmed, rather than a newer helper this note did not verify.
class SessionsController < ApplicationController
  allow_unauthenticated_access only: %i[new create]

  def new
  end

  def create
    user = User.find_by(email_address: params[:email_address]&.strip&.downcase)

    if user&.authenticate(params[:password])
      start_new_session_for(user)
      redirect_to after_authentication_url, notice: "Влязохте успешно."
    else
      redirect_to new_session_path, alert: "Невалиден имейл или парола."
    end
  end

  def destroy
    terminate_session
    redirect_to new_session_path, notice: "Излязохте успешно."
  end
end
