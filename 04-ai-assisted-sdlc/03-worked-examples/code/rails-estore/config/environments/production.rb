# config/environments/production.rb — NOT the path this addendum's docker-compose setup exercises
# (the compose file runs RAILS_ENV=development for a zero-config "friend's first run" — see
# docker-compose.yml). Included for completeness (a real `rails new` app always has this file, the
# gates — rubocop/brakeman — scan it, and a reader deploying for real needs a starting point), and
# because config/application.rb requires SOME environment file to exist for every value RAILS_ENV
# could take. NOTE-SDLC-4-ADD2-docker.md §4.
Rails.application.configure do
  # Code is not reloaded between requests.
  config.enable_reloading = false

  # Eager load code on boot for better performance and memory savings (ignored by cache classes).
  config.eager_load = true

  # Full error reports are disabled; only fully sanitized page renders.
  config.consider_all_requests_local = false
  config.action_controller.perform_caching = true

  # Set to :info for production; :debug lets you see everything, including SQL.
  config.log_level = ENV.fetch("RAILS_LOG_LEVEL", "info")

  # Prepend all log lines with the following tags.
  config.log_tags = [:request_id]

  # "info" includes generic and useful information about system operation, but avoids logging too
  # much information to avoid inadvertent exposure of personally identifiable information (PII).
  config.active_support.report_deprecations = false

  # Use a real queuing backend for Active Job (when this project adds one) instead of async, which
  # in production loses jobs on restart.

  # Enable locale fallbacks for I18n (makes lookups for any locale fall back to the default locale
  # when a translation cannot be found).
  config.i18n.fallbacks = true

  # Don't log any deprecations.
  config.active_record.dump_schema_after_migration = false

  # Only use :id for inspections in production.
  config.active_record.attributes_for_inspect = [:id]

  # Log to STDOUT, the container-native place — the platform (Docker, Kubernetes, a PaaS) captures
  # it from there rather than this app managing a log file at all.
  config.logger = ActiveSupport::Logger.new(STDOUT)
    .tap { |logger| logger.formatter = ::Logger::Formatter.new }
    .then { |logger| ActiveSupport::TaggedLogging.new(logger) }

  # A production deploy behind a TLS-terminating load balancer would leave this off; an
  # internet-facing single-container deploy should uncomment it.
  # config.force_ssl = true
end
