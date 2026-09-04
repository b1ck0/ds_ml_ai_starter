require "rails_helper"

# Covers FEATURE-1 AC1 and AC3. AC3 is the mass-assignment security case: it must fail RED against
# a controller whose `permit()` list includes `:admin`, and PASS once it's removed — that is the
# concrete, testable meaning of "strong parameters block privilege escalation".
RSpec.describe "Registrations", type: :request do
  describe "POST /registration" do
    it "creates a user and signs them in (FEATURE-1 AC1)" do
      post registration_path, params: {
        user: { email_address: "new@example.com", password: "password1", password_confirmation: "password1" }
      }

      expect(response).to redirect_to(root_path)
      expect(User.find_by(email_address: "new@example.com")).to be_present
      expect(Session.last.user.email_address).to eq("new@example.com")
    end

    it "ignores an unpermitted admin param — mass assignment is blocked (FEATURE-1 AC3)" do
      post registration_path, params: {
        user: {
          email_address: "attacker@example.com",
          password: "password1",
          password_confirmation: "password1",
          admin: "true"
        }
      }

      user = User.find_by(email_address: "attacker@example.com")
      expect(user).to be_present
      expect(user.admin).to eq(false)
    end
  end
end
