# A hard assignment, not `||=`: the docker-compose image sets RAILS_ENV=development as its default
# (so `docker compose up` needs zero configuration — see Dockerfile/docker-compose.yml), and that
# container-level env var would otherwise win over `||=` here, silently running the ENTIRE suite
# against config/environments/development.rb instead of test.rb. Discovered by actually running
# `docker compose run --rm web bundle exec rspec` (SPEC-SDLC-4-ADDENDUM-2): every request spec
# failed with a real 403 "Blocked hosts: www.example.com" — development.rb's host allow-list, not
# test.rb's, because RAILS_ENV was never really "test". A test suite must never run against a
# non-test environment no matter what the ambient shell/container already has set.
ENV["RAILS_ENV"] = "test"
require_relative "../config/environment"
abort("The Rails environment is running in production mode!") if Rails.env.production?
require "rspec/rails"

begin
  ActiveRecord::Migration.maintain_test_schema!
rescue ActiveRecord::PendingMigrationError => e
  abort e.to_s.strip
end

RSpec.configure do |config|
  # `fixture_path=` (singular) was rspec-rails' pre-8.0 API; 8.0.4 (this project's pin, per
  # NOTE-SDLC-4-1-versions.md) only defines the plural `fixture_paths=`, taking an array — verified
  # by running the real suite in Docker (SPEC-SDLC-4-ADDENDUM-2) and reading
  # rspec-rails-8.0.4/lib/rspec/rails/configuration.rb, which registers `:fixture_paths` via
  # `config.add_setting` and no longer defines the singular form at all.
  config.fixture_paths = [Rails.root.join("spec/fixtures")]
  config.use_transactional_fixtures = true
  config.infer_spec_type_from_file_location!
  config.filter_rails_from_backtrace!
end
