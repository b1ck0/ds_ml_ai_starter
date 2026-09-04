# Dockerfile — rails-estore, dev-friendly image for `docker compose up`.
# Base image tag confirmed present on Docker Hub (docker-library/ruby's slim-bookworm variant for
# Ruby 4.0.6) — NOTE-SDLC-4-ADD2-docker.md §1.
FROM ruby:4.0.6-slim-bookworm

# build-essential (gcc/make, for compiling native-extension gems) + libsqlite3-dev (sqlite3 gem's
# headers) are build-time-only deps for `bundle install`; no Node/yarn anywhere — Rails 8.1's
# default asset stack (Propshaft + import maps) renders ERB with pure Ruby.
# NOTE-SDLC-4-ADD2-docker.md §3, §5.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends build-essential libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /rails

# Copy just the two files bundle install needs first so this layer is cached across rebuilds that
# only touch application code — the same reasoning as copying a Maven pom.xml before `mvn package`
# so `dependency:go-offline` gets its own cache layer.
COPY Gemfile Gemfile.lock* ./
RUN bundle install

COPY . .

# bin/rails and bin/docker-entrypoint need their executable bit for the native macOS path (README
# §2) and for anyone running this image without the live-edit bind mount (e.g. `docker build` +
# `docker run` directly, or CI). The docker-compose path itself does not depend on this — see
# bin/docker-entrypoint's own comment on why.
RUN chmod +x bin/rails bin/setup bin/docker-entrypoint

ENV RAILS_ENV="development" \
    RAILS_LOG_TO_STDOUT="1"

EXPOSE 3000

ENTRYPOINT ["bash", "bin/docker-entrypoint"]

# Rails' own server command defaults to binding "localhost" (127.0.0.1 + ::1) unless told
# otherwise — config/puma.rb's `bind` directive alone does not override that when going through
# `rails server` (verified: without `-b`, Puma logged "Listening on http://127.0.0.1:3000" from
# inside this exact container, unreachable from the host despite the port mapping). `-b 0.0.0.0`
# is the standard, explicit fix.
CMD ["bundle", "exec", "rails", "server", "-b", "0.0.0.0"]
