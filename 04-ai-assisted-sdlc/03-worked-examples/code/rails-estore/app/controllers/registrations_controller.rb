# RegistrationsController — sign-up. The Rails 8 auth generator does not ship this by default
# (NOTE-SDLC-4-2-auth-generator.md, "Caveats") — FEATURE-1 adds it.
#
# `user_params` is the whole mass-assignment defence in one line: only `email_address`, `password`,
# and `password_confirmation` are ever readable from `params[:user]`. There is no `:admin` in that
# list — not because a form field for it doesn't exist (an attacker doesn't need one; a raw POST body
# with `user[admin]=true` costs nothing to send), but because `permit()` throws away every key that
# isn't named here before `User.new` ever sees the hash. FEATURE-1 AC3's RSpec case
# (spec/requests/registrations_spec.rb) proves this directly rather than trusting the comment.
class RegistrationsController < ApplicationController
  allow_unauthenticated_access

  def new
    @user = User.new
  end

  def create
    @user = User.new(user_params)

    if @user.save
      start_new_session_for(@user)
      redirect_to root_path, notice: "Добре дошли! Профилът ви беше създаден."
    else
      render :new, status: :unprocessable_entity
    end
  end

  private

  def user_params
    params.require(:user).permit(:email_address, :password, :password_confirmation)
  end
end
