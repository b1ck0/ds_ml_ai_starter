# config/boot.rb — first file Rails ever loads (config/environment.rb requires application.rb,
# which requires this). Its ONE job is getting Bundler set up so every subsequent `require` resolves
# against exactly the gems (and versions) this project's Gemfile.lock pins — nothing else may run
# before this. Standard `rails new` content; NOTE-SDLC-4-ADD2-docker.md §4 (boot-file chain:
# Gemfile -> config/boot.rb -> config/application.rb -> ... -> config.ru).
ENV["BUNDLE_GEMFILE"] ||= File.expand_path("../Gemfile", __dir__)

require "bundler/setup" # Set up gems listed in the Gemfile.

# bootsnap caches the (slow) work of parsing/compiling Ruby and YAML so the SECOND boot is much
# faster than the first — a JVM analogy: a bit like a warmed-up JIT cache, except it's a disk cache
# keyed by file content hash, not a runtime tier. Safe to delete (tmp/cache/bootsnap*); it just gets
# rebuilt.
require "bootsnap/setup"
