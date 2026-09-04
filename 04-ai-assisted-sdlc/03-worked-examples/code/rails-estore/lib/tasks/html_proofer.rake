# lib/tasks/html_proofer.rake — FEATURE-3's frontend gate: render the catalog's two page templates
# to static files, then run html-proofer 5.2.2 against them (HTML validity + broken internal links +
# missing alt attributes). Programmatic usage (`HTMLProofer.check_directory(...).run`) matches
# NOTE-SDLC-4-ADD-1-gem-npm-versions.md's cited example exactly. Requires `bundle install` (the
# html-proofer gem lives in the :test group) and a prepared test database
# (`RAILS_ENV=test bin/rails db:test:prepare`).
#
# A real CI run would also want to configure how html-proofer treats the placehold.co external image
# URLs the seed data uses (retry/timeout/allow-list behaviour) -- that is html-proofer's own
# documented option surface (see its README), not something this addendum asserts a flag name for.
namespace :html_proofer do
  desc "Render the product catalog pages and run html-proofer against them"
  task check: :environment do
    require "html-proofer"
    require "action_dispatch/testing/integration"

    out_dir = Rails.root.join("tmp/html_proofer")
    FileUtils.rm_rf(out_dir)
    FileUtils.mkdir_p(out_dir)

    product = Product.first || Product.create!(
      name: "Rails Mug",
      description: "A mug for Rubyists.",
      price_cents: 1_500,
      sku: "MUG-001",
      image_url: "https://placehold.co/600x600?text=Rails+Mug"
    )

    session = ActionDispatch::Integration::Session.new(Rails.application)
    { "index.html" => "/products", "show.html" => "/products/#{product.id}" }.each do |file, path|
      session.get(path)
      File.write(out_dir.join(file), session.response.body)
    end

    HTMLProofer.check_directory(out_dir.to_s).run
  end
end
