# config.ru — the Rack entrypoint. Puma (and any other Rack server) loads THIS file, not
# config/environment.rb directly; `rackup`/Puma's convention is to look for config.ru in the current
# directory. Two lines, standard `rails new` content. NOTE-SDLC-4-ADD2-docker.md §4.
require_relative "config/environment"

run Rails.application
Rails.application.load_server
