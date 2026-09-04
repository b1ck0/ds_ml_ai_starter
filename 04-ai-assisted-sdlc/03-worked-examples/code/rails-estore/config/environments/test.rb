# config/environments/test.rb — loaded only when RAILS_ENV=test, i.e. every `bundle exec rspec` run
# (spec/rails_helper.rb sets `ENV["RAILS_ENV"] ||= "test"` before requiring config/environment).
# NOTE-SDLC-4-ADD2-docker.md §4.
Rails.application.configure do
  # While tests run files are not watched, reloading is not necessary.
  config.enable_reloading = false

  # Eager loading loads your entire application. When running a single test locally this is usually
  # not necessary, and can slow down your test suite.
  config.eager_load = ENV["CI"].present?

  # Show full error reports.
  config.consider_all_requests_local = true
  config.action_controller.perform_caching = false
  config.cache_store = :null_store

  # Render exception templates for rescuable exceptions and raise for other exceptions.
  config.action_dispatch.show_exceptions = :rescuable

  # Disable request forgery protection in test environment — matches Rails' own generated test.rb
  # exactly (confirmed by reading railties-8.1.3.1's app template directly). Without it, every
  # `post`/`patch`/`delete` request spec fails with ActionController::InvalidAuthenticityToken,
  # since request specs don't carry a real CSRF token the way a browser session would.
  config.action_controller.allow_forgery_protection = false

  # Print deprecation notices to stderr.
  config.active_support.deprecation = :stderr

  # Raise error when a before_action's only/except options reference missing actions.
  config.action_controller.raise_on_missing_callback_actions = true

  # ActionDispatch::HostAuthorization otherwise blocks every request spec: Rails' integration test
  # session sends `Host: www.example.com` by default (`ActionDispatch::Integration::Session`'s own
  # hardcoded default host), and it isn't on this app's allow-list — verified by actually running
  # the suite in Docker (SPEC-SDLC-4-ADDENDUM-2): every request spec failed with a real 403 "Blocked
  # hosts: www.example.com" page, not the app's own logic.
  #
  # `config.hosts.clear` looks like the fix but is a trap: when the list is EMPTY, Rails' own
  # railtie re-populates it with a convenience default (.localhost, .test, 0.0.0.0/0, ::/0) that
  # does NOT include www.example.com either — clearing it changes nothing observable (confirmed by
  # printing `config.hosts` after calling `.clear`: still non-empty, still missing this host). The
  # fix that actually works is to add the one host request specs use.
  config.hosts << "www.example.com"
end
