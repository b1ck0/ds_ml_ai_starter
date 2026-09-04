Rails.application.routes.draw do
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
