# config/puma.rb — Puma is Rails 8's default application server (started by `bin/rails server` /
# `rails server`). NOTE-SDLC-4-ADD2-docker.md §4.
max_threads_count = ENV.fetch("RAILS_MAX_THREADS", 3)
min_threads_count = ENV.fetch("RAILS_MIN_THREADS", max_threads_count)
threads min_threads_count, max_threads_count

# Bind explicitly to 0.0.0.0, not just a port — inside a container, binding to the default
# loopback-only address would make the server unreachable from outside the container even with
# `docker compose`'s `ports: ["3000:3000"]` mapping in place. This is the one line that makes
# `docker compose up` actually reachable at http://localhost:3000 from the host.
port ENV.fetch("PORT", 3000)
bind "tcp://0.0.0.0:#{ENV.fetch('PORT', 3000)}"

environment ENV.fetch("RAILS_ENV", "development")

pidfile ENV["PIDFILE"] if ENV["PIDFILE"]

workers ENV.fetch("WEB_CONCURRENCY", 0)
preload_app! if ENV.fetch("WEB_CONCURRENCY", 0).to_i > 0

# Allow puma to be restarted by `bin/rails restart` command.
plugin :tmp_restart

# Run the Solid Queue supervisor inside of Puma for single-server deployments (not used by this
# project — no ActiveJob/Solid Queue gem loaded — left out entirely rather than referencing a
# plugin that isn't in the Gemfile).
