# config/application.rb — declares which Rails frameworks this app boots, then defines the
# Application class every environment file (config/environments/*.rb) reopens with `configure do`.
# Deliberately NOT `require "rails/all"`: this is a small e-store with no mailers, background jobs,
# WebSockets, or file uploads, so only the frameworks the code actually uses are loaded — one fewer
# unused subsystem to reason about when something goes wrong, the same instinct as a lean Maven
# `<dependencies>` list over a fat "include everything" parent POM. Propshaft and importmap-rails
# (the Gemfile) don't need an explicit require here: `Bundler.require` below loads every gem in the
# Gemfile's default + current-environment groups, and each one registers its own Railtie on require.
# NOTE-SDLC-4-ADD2-docker.md §3, §4.
require_relative "boot"

require "rails"
require "active_model/railtie"
require "active_record/railtie"
require "action_controller/railtie"
require "action_view/railtie"

# Require the gems listed in Gemfile, including any gems you've limited to :test, :development, or
# :production.
Bundler.require(*Rails.groups)

module RailsEstore
  class Application < Rails::Application
    # Initialize configuration defaults for the originally generated Rails version. Every new
    # default Rails ships after this version is opted into explicitly, not silently inherited — the
    # same reason a Java project pins a language level (`<maven.compiler.release>`) instead of
    # floating with whatever javac happens to default to.
    config.load_defaults 8.1

    # Autoload lib/ (Zeitwerk, Rails' own class loader) but skip lib/tasks — .rake files are Rake
    # DSL, not Ruby classes/modules, and Zeitwerk would otherwise expect a matching constant for
    # every file it finds there and raise on boot.
    config.autoload_lib(ignore: %w[assets tasks])

    # Please, see config/environments/*.rb for env-specific configuration.
  end
end
