Rails.application.routes.draw do
  # Rails 8's health-check endpoint — routed explicitly, not automatic. `Rails::HealthController`
  # ships with the framework (returns 200 if the app booted with no exceptions, 500 otherwise; it
  # does NOT check the database or any other dependency), but a fresh `rails new` app is the one
  # that WIRES this route by default; this project's routes.rb predates that scaffolding, so this
  # addendum adds the line itself. NOTE-SDLC-4-ADD2-docker.md §3. Docker health-check target:
  # `curl -f http://localhost:3000/up` — see docker-compose.yml / README.md.
  get "up" => "rails/health#show", as: :rails_health_check

  root "products#index"

  # Rails 8 native auth generator convention — a singular `resource`, not `resources`: there is only
  # ever one "current session" from the browser's point of view. NOTE-SDLC-4-2-auth-generator.md.
  resource :session, only: %i[new create destroy]
  resource :registration, only: %i[new create]

  resources :products, only: %i[index show]
  resource :cart, only: :show
  resources :line_items, only: %i[create destroy]

  namespace :checkout do
    resources :orders, only: %i[new create show]
  end
end
