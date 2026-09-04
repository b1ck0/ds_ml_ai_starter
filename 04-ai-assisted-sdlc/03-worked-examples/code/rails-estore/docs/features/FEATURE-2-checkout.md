# FEATURE-2: От количка до поръчка — checkout с фиктивен (stubbed) платежен интерфейс

**Status:** approved
**Owner:** архитект (Opus)
**Routing:** implementer=Sonnet · research=NOTE-SDLC-4-1, NOTE-SDLC-4-4 · review=Sonnet (fresh)

## Намерение (Intent)

Изписан (signed-in) потребител (FEATURE-1) може да добавя продукти в количка и да прави checkout:
редовете на количката стават редове на `Order`, поръчката преминава през enum за статус
(`pending → processing → completed/cancelled`), и интерфейс `PaymentService` извиква
`Stripe::Checkout::Session` в тестов режим — реална точка на интеграция, фиктивно (stubbed)
таксуване, без жив ключ, без нужда от Stripe акаунт за пускане на тестовия пакет
[source: NOTE-SDLC-4-4-stripe-checkout.md]. Всяко търсене на поръчка е ограничено до изписания
потребител: това е функционалността, при която липсваща проверка за авторизация нанася най-много
щети — неограничен (unscoped) `Order.find(params[:id])` позволява на всеки изписан потребител да
прочете (и, ако действието го позволява, да действа върху) чужда поръчка, само с гадаене на малко
цяло число.

## Критерии за приемане (Acceptance criteria)

- AC1 — `POST /checkout/orders` с непразна количка създава една `Order` (`status: pending`, после
  `processing`, щом `PaymentService` върне сесия), по един `LineItem` на отделен продукт в количката
  (копиран, не преместен, докато транзакцията не се commit-не), и изпразва количката.
- AC2 — `POST /checkout/orders` с ПРАЗНА количка не създава `Order` и пренасочва обратно към
  количката с предупреждение (alert).
- AC3 — `Order#status` е едно от `pending`, `processing`, `completed`, `cancelled` (Rails `enum`, не
  колона със свободен текст) [source: NOTE-SDLC-4-4-stripe-checkout.md].
- AC4 — `PaymentService#create_checkout_session`, когато `Rails.env.test?` (или `STRIPE_STUB=true`),
  връща фиктивен (stub) session hash и НЕ прави мрежово извикване — верифицирано чрез stub-ване/
  mock-ване на `Stripe::Checkout::Session.create` и потвърждение, че никога не е извикан по време на
  тестовия път [source: NOTE-SDLC-4-4-stripe-checkout.md].
- AC5 — `GET /checkout/orders/:id` връща поръчката, когато принадлежи на изписания потребител, и
  връща 404 (чрез `Current.user.orders.find`, който хвърля `ActiveRecord::RecordNotFound` за id извън
  собствените поръчки на този потребител — не гол `Order.find`), когато принадлежи на ДРУГ
  потребител. Това е случаят за авторизация; не се удовлетворява само от "маршрутът изисква вход".
- AC6 — повтарящ се `POST /checkout/orders` за същата количка след първи успешен checkout не води
  тихомълком до двойно таксуване: количката е празна след AC1, така че второ подаване удря пътя на
  празна количка от AC2.

## Твърдения за заземяване (Claims to ground)

- Формата на асоциацията Order/LineItem, идиомата `enum status:` — заземено,
  `docs/research/NOTE-SDLC-4-4-stripe-checkout.md`.
- Версия на Stripe gem-а, `Stripe::Checkout::Session` като текущата точка на интеграция за hosted
  checkout, префикси на тестови ключове, източник на credentials — заземено,
  `docs/research/NOTE-SDLC-4-4-stripe-checkout.md`.
- Категориите проверки на Brakeman за mass-assignment/SQLi/XSS, използвани при review на
  controller-а на тази функционалност — заземено, `docs/research/NOTE-SDLC-4-3-brakeman-checks.md`.

## Извън обхват (Out of scope)

- Реален Stripe webhook, обработващ асинхронно потвърждение на плащане — интерфейсът е фиктивен
  (stubbed); окабеляването на реален webhook е следваща функционалност, щом съществува Stripe
  акаунт.
- Възстановявания, частична отмяна, намаляване на наличности (inventory/stock) — не са нужни за
  демонстрация на управлявания цикъл; всяко от тях би било собствена feature spec.
- Персонализирана форма за плащане със Stripe Elements — hosted Checkout
  (`Stripe::Checkout::Session`) е по-просто и достатъчно тук
  [source: NOTE-SDLC-4-4-stripe-checkout.md, "Caveats"].

## Активи за изготвяне (Assets to produce)

- `app/models/product.rb`, `app/models/cart.rb`, `app/models/line_item.rb`, `app/models/order.rb`
- `app/controllers/products_controller.rb`, `app/controllers/carts_controller.rb`,
  `app/controllers/line_items_controller.rb`, `app/controllers/checkout/orders_controller.rb`
- `app/services/payment_service.rb`
- `app/views/products/*`, `app/views/carts/show.html.erb`, `app/views/checkout/orders/*`
- допълнения към `db/schema.rb` (таблиците products, carts, line_items, orders)
- `spec/models/order_spec.rb`, `spec/requests/checkout_spec.rb`

## Гейтове (Gates)

Вход: тази спецификация одобрена; FEATURE-1 мержнат (checkout изисква изписан потребител); двете
бележки за заземяване по-горе пристигнали. Изход: чеклистът `docs/definition-of-done.md`, изцяло —
включително случая за авторизация от AC5, с изричен RSpec пример (втори потребител, опитващ се да
прочете поръчката на първия потребител).
