# NOTE-SDLC-4-5: has_secure_password uses bcrypt; strong parameters prevent mass assignment

**Answer:**

**`has_secure_password` uses bcrypt:** The User model calls `has_secure_password`, which automatically:
- Hashes the plaintext `password` using bcrypt (deliberate CPU-intensive algorithm, ~200–300ms per hash to resist brute force)
- Stores the hash in a `password_digest` column (never the plaintext)
- Validates `password` and `password_confirmation` match
- Provides `authenticate(plaintext)` method to verify login

**Strong parameters prevent mass assignment:** Without explicit `permit()`, all parameter attributes are forbidden in mass-assignment contexts (e.g., `User.create(params[:user])`). Only explicitly whitelisted attributes (via `params.require(:user).permit(:email, :password, :password_confirmation)`) are passed to the model. This blocks attackers from injecting `role: "admin"` or `is_verified: true` via form parameters.

**Evidence:**

1. **has_secure_password uses bcrypt** — [ActiveModel::SecurePassword::ClassMethods - Rails API](https://api.rubyonrails.org/classes/ActiveModel/SecurePassword/ClassMethods.html) official Rails API docs: "SecurePassword uses bcrypt to hash passwords... Hashing is performed with a deliberately slow algorithm (200–300ms) to make brute-force attacks impractical."

2. **password_digest column required** — Rails API docs state: "Adds `has_secure_password` to your User model. This requires a migration to add a `password_digest` column to the users table (string, null: false)."

3. **bcrypt hashing and salting** — [Secure Passwords With Rails And Bcrypt](https://til.hashrocket.com/posts/kslhwk22ik-secure-passwords-with-rails-and-bcrypt) (HashrocketTIL): "bcrypt is a deliberately slow hashing algorithm... includes automatic salting and makes brute-force attacks impractical."

4. **Mass assignment vulnerability** — [Brakeman - Mass Assignment](https://brakemanscanner.org/docs/warning_types/mass_assignment/): "Mass Assignment occurs when an attacker is able to set any attribute of your model, including those not intended for mass assignment. Without Strong Parameters, a malicious user could add `user[admin]=true` to a form submission and grant themselves admin access."

5. **Strong parameters protect via explicit permit()** — [Rails Strong Parameters Deep Dive](https://blog.saeloun.com/2025/02/18/deep-dive-into-rails-action-controller-strong-parameters/): "With strong parameters, Action Controller parameters are forbidden to be used in Active Model mass assignments until they have been permitted... params.require(:user).permit(:name, :email, :password, :password_confirmation) explicitly whitelists allowed attributes."

6. **Strong parameters in Rails 4+** — [Common Rails Security Pitfalls and Their Solutions](https://www.sitepoint.com/common-rails-security-pitfalls-and-their-solutions/) and [Rails security guide](https://guides.rubyonrails.org/security.html) confirm strong parameters are the standard protection mechanism for mass assignment in Rails 4 and later (Rails 5+ has it on by default).

7. **Rails 8 authentication generator uses strong parameters** — [Rails 8 adds built in authentication](https://blog.saeloun.com/2025/05/12/rails-8-adds-built-in-authentication-generator/) confirms the generated registration controller uses `params.require(:user).permit(...)` to safely extract and create users.

**Caveats / limits:**

- **bcrypt cost factor:** The default cost factor (12) can be tuned if performance is a concern (higher = more secure but slower). Rails' `has_secure_password` uses the default; no manual tuning needed for a teaching app.
- **password_confirmation:** `has_secure_password` validates that `password` and `password_confirmation` match *before* hashing. This is a UX check only; the confirmation value is never stored or hashed.
- **Strong parameters are not authorization:** Even with strong parameters, a user can modify their own email or password. Authorization checks (e.g., `before_action :require_current_user_ownership` in controller) are still needed to prevent a user from editing another user's record.
- **Never log or inspect password fields:** Even though the password is hashed before storage, never log `params[:user]` or inspect `user.password` in debug output, as the plaintext could be exposed in logs.
- Date checked: **2026-09-04**.

**Recommendation:**

1. **User model setup:**
   ```ruby
   class User < ApplicationRecord
     has_secure_password
     
     validates :email_address, presence: true, uniqueness: true
     validates :password, length: { minimum: 8 }, if: -> { password.present? }
   end
   ```
   Requires migration:
   ```ruby
   class CreateUsers < ActiveRecord::Migration[8.1]
     def change
       create_table :users do |t|
         t.string :email_address, null: false
         t.string :password_digest, null: false
         t.timestamps
       end
       
       add_index :users, :email_address, unique: true
     end
   end
   ```

2. **Registration controller (strong parameters in action):**
   ```ruby
   class RegistrationsController < ApplicationController
     def create
       # Whitelist only email, password, password_confirmation
       user_params = params.require(:user).permit(:email_address, :password, :password_confirmation)
       user = User.new(user_params)
       
       if user.save
         session = user.sessions.create!(ip_address: request.ip, user_agent: request.user_agent)
         cookies.signed.permanent[:session_token] = { value: session.token }
         redirect_to root_path, notice: "Signed up successfully"
       else
         render :new, status: :unprocessable_entity
       end
     end
   end
   ```

3. **Test the security gates:**
   - **Attempt mass assignment:** In a test, try `User.create(email_address: "test@example.com", password: "secret", admin: true)`. Without `permit(:admin)`, the `admin` attribute should be silently ignored and not set.
   - **Verify password hashing:** Login with plaintext password; verify it never matches the stored `password_digest` value directly (use `user.authenticate(plaintext)` method instead).
   - **Brakeman detects mass assignment gaps:** If a controller forgets `permit()`, Brakeman will flag it as a mass-assignment vulnerability.

This teaches the reader that `has_secure_password` + `strong parameters` + authorization checks form a layered defense against a category of common Rails vulnerabilities.
