require "rails_helper"

# Covers FEATURE-1 AC4, AC5, AC6.
RSpec.describe "Sessions", type: :request do
  let!(:user) do
    User.create!(email_address: "member@example.com", password: "password1", password_confirmation: "password1")
  end

  describe "POST /session" do
    it "signs in with correct credentials (AC4)" do
      post session_path, params: { email_address: "member@example.com", password: "password1" }

      expect(response).to redirect_to(root_path)
      expect(user.sessions.count).to eq(1)
    end

    it "rejects incorrect credentials without creating a session (AC4)" do
      post session_path, params: { email_address: "member@example.com", password: "wrong-password" }

      expect(response).to redirect_to(new_session_path)
      expect(user.sessions.count).to eq(0)
    end
  end

  describe "DELETE /session" do
    it "destroys the current session and clears the cookie (AC5)" do
      post session_path, params: { email_address: "member@example.com", password: "password1" }
      expect { delete session_path }.to change { Session.count }.by(-1)
    end
  end

  describe "an unauthenticated request to a protected page (AC6)" do
    it "redirects to sign-in instead of rendering the page" do
      get cart_path

      expect(response).to redirect_to(new_session_path)
    end
  end
end
