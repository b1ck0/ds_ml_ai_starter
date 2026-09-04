# config/environment.rb — last link in the boot chain. Loading config/application.rb only DEFINES
# the Application class; `initialize!` is the call that actually runs every initializer (including
# each config/environments/<env>.rb file) and freezes the app's configuration. config.ru requires
# this file, and only this file, to bring up the whole app.
require_relative "application"

Rails.application.initialize!
