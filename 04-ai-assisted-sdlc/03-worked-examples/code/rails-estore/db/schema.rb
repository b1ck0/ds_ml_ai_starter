# This file is auto-generated from the current state of the database by `bin/rails db:schema:dump`
# in a real app; it is hand-written here to give the models above a concrete, correct table shape to
# run against, since no Ruby toolchain runs in this book's own repository (see the chapter's
# Environment note). Column choices are grounded: `password_digest` per NOTE-SDLC-4-5-auth-security.md;
# `status` as an integer enum column and the Order/LineItem shape per
# NOTE-SDLC-4-4-stripe-checkout.md.

ActiveRecord::Schema[8.1].define(version: 2026_09_04_000_006) do
  create_table "users", force: :cascade do |t|
    t.string "email_address", null: false
    t.string "password_digest", null: false
    t.boolean "admin", default: false, null: false
    t.timestamps
    t.index ["email_address"], name: "index_users_on_email_address", unique: true
  end

  create_table "sessions", force: :cascade do |t|
    t.integer "user_id", null: false
    t.string "token", null: false
    t.string "ip_address"
    t.string "user_agent"
    t.timestamps
    t.index ["token"], name: "index_sessions_on_token", unique: true
    t.index ["user_id"], name: "index_sessions_on_user_id"
  end

  create_table "products", force: :cascade do |t|
    t.string "name", null: false
    t.text "description"
    t.integer "price_cents", null: false
    t.timestamps
  end

  create_table "carts", force: :cascade do |t|
    t.integer "user_id", null: false
    t.timestamps
    t.index ["user_id"], name: "index_carts_on_user_id", unique: true
  end

  create_table "orders", force: :cascade do |t|
    t.integer "user_id", null: false
    t.integer "status", default: 0, null: false
    t.string "stripe_session_id"
    t.timestamps
    t.index ["stripe_session_id"], name: "index_orders_on_stripe_session_id", unique: true
    t.index ["user_id"], name: "index_orders_on_user_id"
  end

  create_table "line_items", force: :cascade do |t|
    t.integer "product_id", null: false
    t.string "cartable_type", null: false
    t.integer "cartable_id", null: false
    t.integer "quantity", default: 1, null: false
    t.timestamps
    t.index ["cartable_type", "cartable_id"], name: "index_line_items_on_cartable"
    t.index ["product_id"], name: "index_line_items_on_product_id"
  end

  add_foreign_key "carts", "users"
  add_foreign_key "line_items", "products"
  add_foreign_key "orders", "users"
  add_foreign_key "sessions", "users"
end
