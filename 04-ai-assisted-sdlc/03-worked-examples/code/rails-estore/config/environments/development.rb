# config/environments/development.rb — loaded only when RAILS_ENV=development (the default when no
# RAILS_ENV is set at all, which is how this project's docker-compose.yml runs it: no asset
# precompile step, no SECRET_KEY_BASE/master key needed, and code changes under the bind-mounted
# app/ directory take effect on the next request with no restart — the "friend's first run" path
# this addendum optimizes for). NOTE-SDLC-4-ADD2-docker.md §3, §4.
Rails.application.configure do
  # In the development environment your application's code is reloaded any time it changes without
  # you having to restart the server.
  config.enable_reloading = true

  # Do not eager load code on boot.
  config.eager_load = false

  # Show full error reports.
  config.consider_all_requests_local = true

  # Don't bother caching in dev — controller responses are cheap, and caching would hide a change
  # you just made from the very page you're looking at.
  config.action_controller.perform_caching = false

  # Print deprecation notices to the Rails logger.
  config.active_support.deprecation = :log

  # Raise an error on page load if there are pending migrations. This project has no db/migrate
  # directory (schema.rb is loaded directly by db:prepare), so this never fires — kept in because
  # it's the stock `rails new` default and costs nothing to leave on.
  config.active_record.migration_error = :page_load

  # Log the SQL a request issues, not just the request itself — useful while learning the app's
  # data access patterns.
  config.active_record.verbose_query_logs = true
end
