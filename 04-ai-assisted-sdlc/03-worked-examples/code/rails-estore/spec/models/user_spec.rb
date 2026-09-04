require "rails_helper"

# Covers FEATURE-1 AC2 (password hashing) and the model-level half of AC3 (mass assignment is a
# controller-layer defence, but the model's own attribute list matters too — `admin` must be a real
# column with a safe default, not something has_secure_password quietly exposes).
RSpec.describe User, type: :model do
  describe "password hashing (FEATURE-1 AC2)" do
    it "stores a bcrypt digest, never the plaintext password" do
      user = User.create!(email_address: "ada@example.com", password: "s3cret-pw", password_confirmation: "s3cret-pw")

      expect(user.password_digest).not_to eq("s3cret-pw")
      expect(user.password_digest).to be_present
      expect(user.authenticate("s3cret-pw")).to eq(user)
      expect(user.authenticate("wrong-password")).to be false
    end
  end

  describe "validations" do
    it "requires a unique, present email address" do
      User.create!(email_address: "dup@example.com", password: "password1", password_confirmation: "password1")
      dup = User.new(email_address: "dup@example.com", password: "password1", password_confirmation: "password1")

      expect(dup).not_to be_valid
      expect(dup.errors[:email_address]).to include("has already been taken")
    end

    it "normalizes email_address to lowercase, stripped" do
      user = User.create!(email_address: "  Ada@Example.com  ", password: "password1", password_confirmation: "password1")

      expect(user.email_address).to eq("ada@example.com")
    end
  end

  it "defaults admin to false for every new user" do
    user = User.create!(email_address: "grace@example.com", password: "password1", password_confirmation: "password1")

    expect(user.admin).to eq(false)
  end
end
