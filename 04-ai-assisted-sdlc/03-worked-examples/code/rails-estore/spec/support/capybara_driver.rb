# spec/support/capybara_driver.rb — registers a headless Chrome driver for the `js: true` system
# specs that need real JS execution: axe-core-capybara injects and runs a JS library against the
# rendered DOM, and Capybara's default driver (rack_test) has no JS engine at all.
#
# Required explicitly from spec/system/accessibility_spec.rb and spec/system/seo_spec.rb rather than
# auto-loaded from spec/rails_helper.rb, which this addendum does not modify
# (SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents: preserve existing files byte-for-byte except the
# named wiring changes -- rails_helper.rb is not one of them). Needs Chrome/Chromium plus a matching
# chromedriver on PATH -- see README.md's "Frontend gate" section.
require "capybara/rspec"
require "selenium-webdriver"

Capybara.register_driver :selenium_chrome_headless do |app|
  options = Selenium::WebDriver::Chrome::Options.new
  options.add_argument("--headless=new")
  options.add_argument("--disable-gpu")
  options.add_argument("--no-sandbox")
  Capybara::Selenium::Driver.new(app, browser: :chrome, options: options)
end

Capybara.javascript_driver = :selenium_chrome_headless

RSpec.configure do |config|
  config.before(:each, type: :system) do
    driven_by Capybara.javascript_driver
  end
end
