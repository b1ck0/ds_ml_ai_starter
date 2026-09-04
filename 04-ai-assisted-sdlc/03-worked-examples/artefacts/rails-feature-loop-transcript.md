# Референтен транскрипт — FEATURE-1 и FEATURE-2 през управлявания цикъл

> **Това е РЕФЕРЕНТЕН транскрипт, не заснет лог от сесия.** Той разказва, стъпка по стъпка, какво
> реално произвежда диспечирането на състава от архитект/researcher/implementer/reviewer в
> `rails-estore` — използвайки реалните файлове, commit-нати в тази глава под
> [`code/rails-estore/`](../code/rails-estore/) — за да видиш формата на цикъла, без да плащаш за
> реално изпълнение. В **това Python хранилище за книгата няма Ruby toolchain** (виж бележката за
> средата в [`03-rails-estore-sdlc.md`](../03-rails-estore-sdlc.md)), затова всеки конзолен блок с
> `bundle exec rspec` / `rubocop` / `brakeman` по-долу е **илюстративен**: имената на файловете,
> спецификацията, кодът и присъдите от review-а са реални и съответстват на commit-натото; точните
> байтове на конзолата от изпълнение не са транскрипт на реално стартиране.

---

## Част А — FEATURE-1: вход на потребител (чист успех от първия път)

### Стъпка 1 — Заявка → спецификация

**Prompt на собственика към архитекта:**
> „Преди да има количка, трябва да има акаунт. Добави регистрация, вход, изход, използвайки формата
> на вградения auth generator на Rails 8, не Devise."

**Архитектът (Opus)**, съгласно златно правило 1 на `CLAUDE.md`, пише
[`docs/features/FEATURE-1-user-login.md`](../code/rails-estore/docs/features/FEATURE-1-user-login.md):
намерение, шест критерия за приемане (AC1–AC6), и секция „твърдения за заземяване", назоваваща три
бележки, на които спецификацията разчита.

### Стъпка 2 — Заземяване на неизвестните

Трите твърдения, които спецификацията цитира, вече бяха заземени преди тази функционалност да бъде
скопирана: `docs/research/NOTE-SDLC-4-2-auth-generator.md` (формата на generator-а и сравнението с
Devise), `docs/research/NOTE-SDLC-4-1-versions.md` (версия/CVE статус на bcrypt), и
`docs/research/NOTE-SDLC-4-5-auth-security.md` (bcrypt хеширането на `has_secure_password` и strong
parameters). Не е нужно ново диспечиране — това е същият модел „стъпката съществува, но е no-op,
когато няма нищо за проверка", който FEATURE-1 в Java главата също показа.

### Стъпка 3 — Първо провалящ се тест, после имплементация

**Архитектът диспечира implementer-а** (`.claude/agents/implementer.md`, Sonnet) срещу
`FEATURE-1-user-login.md`.

**Implementer, стъпка 1 — пише RSpec примерите преди да съществува код, който да ги удовлетворява:**

```
$ implementer writes spec/models/user_spec.rb, spec/requests/registrations_spec.rb,
    spec/requests/sessions_spec.rb
$ implementer writes STUB versions of User, Session, Current, Authentication, and the two
    controllers (models with no validations/has_secure_password yet; controllers that just
    `render plain: "not implemented", status: :not_implemented`)
$ bundle exec rspec spec/models/user_spec.rb spec/requests/registrations_spec.rb spec/requests/sessions_spec.rb
```

```text
[REFERENCE — illustrative RSpec output, not captured in this sandbox]
Failures:

  1) User password hashing (FEATURE-1 AC2) stores a bcrypt digest, never the plaintext password
     NoMethodError: undefined method `authenticate' for #<User:0x00007f2c1a3b0e10>
  2) Registrations POST /registration creates a user and signs them in (FEATURE-1 AC1)
     Failure/Error: expect(response).to redirect_to(root_path)
       expected: a redirect to "http://www.example.com/"
            got: 501 (not implemented)
  3) Registrations POST /registration ignores an unpermitted admin param ...
     Failure/Error: expect(user).to be_present
       expected User(id: nil) to be present, got nil
  ...
8 examples, 8 failures
```

Добре — всеки провал е ОТ ПРАВИЛНИЯ вид (поведението наистина още не съществува: няма
`has_secure_password`, няма контролерна логика), не е печатна грешка в самия spec файл. Точно това
изисква златно правило 2 на `CLAUDE.md` и секцията „Test-first" на `docs/definition-of-done.md`,
преди да бъде написан и един ред реална логика.

**Implementer, стъпка 2 — пише минималния production код:** заменя заглушките (stubs) с реалните
`User`/`Session`/`Current`/`Authentication` и двата реални контролера — точно файловете, commit-нати
под [`code/rails-estore/app/`](../code/rails-estore/app/).

### Стъпка 4 — Гейт

`.claude/hooks/guard.sh` (`PreToolUse`) се изпълняваше тихо при всеки `Bash` извикан от implementer-а
— би блокирал, например, `git commit --no-verify` направо, ако implementer-ът се беше опитал да
заобиколи провалящ се гейт. `verify.sh` (`PostToolUse`) се задействаше след всеки `Edit`/`Write`
върху `.rb` файл:

```text
[REFERENCE — illustrative; a real run needs bundle on PATH]
[verify] ruby gate: app/models/user.rb
  $ bundle exec rspec
  $ bundle exec rubocop
  $ bundle exec brakeman -q --no-summary
[verify] all three gates green
```

Implementer-ът докладва пълния чеклист на `docs/definition-of-done.md` обратно на архитекта,
свързвайки AC1–AC6 с RSpec примерите, които ги покриват:

| AC | RSpec пример |
|---|---|
| AC1 | `Registrations POST /registration creates a user and signs them in` |
| AC2 | `User password hashing stores a bcrypt digest, never the plaintext password` |
| AC3 | `Registrations POST /registration ignores an unpermitted admin param` |
| AC4 | `Sessions POST /session signs in / rejects incorrect credentials` |
| AC5 | `Sessions DELETE /session destroys the current session` |
| AC6 | `Sessions an unauthenticated request to a protected page redirects to sign-in` |

### Стъпка 5 — Review

**Архитектът диспечира ФРЕШ reviewer** (`.claude/agents/reviewer.md`, нов Sonnet контекст, който
никога не е виждал скицовата работа на implementer-а). Той минава през собствения си процес по ред:
съответствие спрямо всеки AC, доказателство за test-first (логът с провалите от стъпка 1 по-горе),
авторизация (`carts_controller.rb` — изобщо няма `params[:id]` lookup, най-простата форма на „не може
да е грешно"), mass assignment (`registrations_controller.rb#user_params` — точно три ключа, без
`:admin`), заземяване (и трите цитирани бележки съществуват и твърдят това, което спецификацията
казва), `sk_live`/`pk_live` grep през diff-а (нищо), после независимо пуска отново
`bundle exec rspec`, `rubocop`, `brakeman -q --no-summary` сам, вместо да се довери на доклада на
implementer-а.

**Присъда на reviewer-а:**
> **ОДОБРЕНО (APPROVE).** AC1–AC6 всеки се свързва с минаващ RSpec пример; редът test-first е
> доказан от лога с провалите от стъпка 1; `rubocop`/`brakeman` са и двата чисти; permit списъкът
> на `registrations_controller.rb` е минимален и не включва `admin`; никой файл извън „Активи за
> произвеждане" на `FEATURE-1` не е променен. Без забележки.

### Стъпка 6 — Merge

**Архитектът прави merge**, съгласно правилото за одобрение на merge на `CLAUDE.md`. Тялото на PR-а
свързва всеки критерий за приемане с доказателството му (таблицата по-горе + лога от гейта).

---

## Част Б — FEATURE-2: checkout (внедрен пропуск в авторизацията, уловен и поправен)

### Стъпка 1 — Заявка → спецификация

**Prompt на собственика към архитекта:**
> „Сега, когато потребител може да влиза, нека добавя продукти в количка и да плаща. Направи заглушка
> (stub) за реалното Stripe таксуване — без акаунт, без реален ключ — но включи реалната интеграционна
> точка."

**Архитектът** пише
[`docs/features/FEATURE-2-checkout.md`](../code/rails-estore/docs/features/FEATURE-2-checkout.md):
шест критерия за приемане (AC1–AC6), изрично обозначавайки AC5 като „случая с авторизацията; не се
удовлетворява само от 'маршрутът изисква вход'" — умишлен флаг на ниво спецификация за точно този
дефект, който тази стъпка предстои да покаже.

### Стъпка 2 — Заземяване на неизвестните

`docs/research/NOTE-SDLC-4-4-stripe-checkout.md` (формата на Order/LineItem, `Stripe::Checkout::Session`,
префиксите на тестовите ключове) и `docs/research/NOTE-SDLC-4-3-brakeman-checks.md` (категориите
проверки на Brakeman — цитирани от reviewer-а по-долу, не от implementer-а) вече бяха приземени; не е
нужно ново диспечиране.

### Стъпка 3 — Първо провалящ се тест, после имплементация (пропускът)

**Implementer, стъпка 1:** пише `spec/models/order_spec.rb` и `spec/requests/checkout_spec.rb`,
потвърждава, че всеки пример се проваля по правилната причина срещу stub контролери, същата
дисциплина като в Част А. Под времеви натиск обаче пише примера за AC5 само като ЕДИН ПОЗИТИВЕН
случай — *„връща поръчката на нейния собствен собственик"* — и никога не пише отрицателния случай,
който собствената формулировка на спецификацията изисква (*„НЕ връща поръчката на друг потребител"*).
Нищо не улавя това в момента на писане: RSpec няма начин да сигнализира липсващ пример, само
провалящ се.

**Implementer, стъпка 2:** пише `Checkout::OrdersController`. Действието `create` е наред — то
изгражда поръчката чрез `Current.user.orders.create!`. Действието `show` излиза с грешката:

```ruby
# app/controllers/checkout/orders_controller.rb — AS FIRST WRITTEN (the planted slip)
def show
  @order = Order.find(params[:id])   # <-- not scoped to Current.user
end
```

Това е учебникарски IDOR (insecure direct object reference, несигурна директна референция към
обект): всеки влязъл потребител може да прочете чужда поръчка, като смени id-то в URL адреса.
Позицията „затворено по подразбиране" на `docs/architecture.md` (§1) се отнася до *автентикация* —
дали изобщо си влязъл — и това действие все още го изисква; грешката е, че влизането като *когото и
да е* се третира като авторизация за четене на поръчката на *когото и да е*.

### Стъпка 4 — Гейт (зелен — и точно това е проблемът)

```text
[REFERENCE — illustrative]
[verify] ruby gate: app/controllers/checkout/orders_controller.rb
  $ bundle exec rspec
  $ bundle exec rubocop
  $ bundle exec brakeman -q --no-summary
[verify] all three gates green
```

И трите гейта докладват чисто, по причина, върху която си струва да се замислим, а не по случайност:

- **RSpec е зелен**, защото липсващият отрицателен пример означава, че нищо не упражнява уязвимия
  път — зелен suite доказва, че съществуващите тестове минават, не че съществуват правилните тестове.
- **RuboCop е зелен**, защото `Order.find(params[:id])` е напълно идиоматичен, добре стилизиран Ruby.
  Стилът няма мнение за това към кой модел е обхванат (scoped) даден query.
- **Brakeman е зелен**, защото счупен контрол на достъпа между записи от *същия* тип не е една от
  неговите статични проверки — собствената секция „Ограничения" на
  `docs/research/NOTE-SDLC-4-3-brakeman-checks.md` казва това направо: Brakeman „няма да улови
  логически грешки (напр. пропуски в авторизацията, които изискват динамичен анализ или тестово
  покритие)." Това е точно тази пролука.

Implementer-ът докладва зеления гейт и предава за review, честно вярвайки, че FEATURE-2 е готова —
докладът не е нечестен, просто измерва грешното нещо.

### Стъпка 5 — Review (ПРОМЕНИ ЗАЯВЕНИ)

**Диспечира се фреш reviewer.** Неговият процес (`.claude/agents/reviewer.md` стъпка 3) изисква
проверка на авторизацията **на ръка, за всяко засегнато действие на контролер** — не извеждане на
заключение от зелен гейт. Четейки директно `Checkout::OrdersController#show`, той открива необхванатия
(unscoped) `Order.find` за времето, нужно да се прочете редът, и кръстосано проверява спрямо
спецификацията: AC5 изрично изисква 404 за чужда поръчка, а никой RSpec пример не го доказва в
никоя посока.

**Присъда на reviewer-а:**
> **ПРОМЕНИ ЗАЯВЕНИ (CHANGES REQUESTED).**
> 1. `app/controllers/checkout/orders_controller.rb:26` — `Order.find(params[:id])` не е обхванат
>    (scoped) към `Current.user`. Всеки влязъл потребител може да прочете чужда поръчка, като
>    увеличава id-то в URL адреса (IDOR / счупен контрол на достъпа, OWASP A01:2021). Това е точно
>    режимът на провал, който AC5 в `docs/features/FEATURE-2-checkout.md` назовава по име — „не се
>    удовлетворява само от 'маршрутът изисква вход'." Поправка: `Current.user.orders.find(params[:id])`.
> 2. `spec/requests/checkout_spec.rb` — отрицателният случай на AC5 (втори потребител се опитва да
>    прочете поръчката на първия) няма RSpec пример. Зеленият RSpec тук не е доказателство за нищо:
>    добави случая, потвърди, че се проваля срещу текущия код, после потвърди, че поправката от
>    констатация 1 го прави да минава.
> И двете констатации блокират merge. Чистотата на RuboCop и Brakeman не е достатъчна — виж
> собствената `docs/research/NOTE-SDLC-4-3-brakeman-checks.md` на този проект, която казва направо, че
> Brakeman не покрива този клас грешки; точно за това служи фреш reviewer-ът.

### Стъпка 3, отново — поправката

**Архитектът пренасочва присъдата обратно към implementer-а** (втората изходна рампа на цикъла,
`docs/architecture.md` §3 стъпка 5 → стъпка 3). Implementer-ът:

1. Добавя липсващия отрицателен RSpec пример към `spec/requests/checkout_spec.rb` — точно примера
   `"does NOT return another user's order — the IDOR case"`, commit-нат в
   [`code/rails-estore/spec/requests/checkout_spec.rb`](../code/rails-estore/spec/requests/checkout_spec.rb)
   — и потвърждава, че се проваля срещу все още счупения контролер:

```text
[REFERENCE — illustrative]
Failures:
  1) Checkout GET /checkout/orders/:id — authorization (AC5) does NOT return another user's order — the IDOR case
     Failure/Error: get checkout_order_path(others_order)
       expected ActiveRecord::RecordNotFound but nothing was raised
1 example, 1 failure
```

2. Прилага еднолинейната поправка: `Order.find(params[:id])` → `Current.user.orders.find(params[:id])`
   (версията, commit-ната в
   [`app/controllers/checkout/orders_controller.rb`](../code/rails-estore/app/controllers/checkout/orders_controller.rb)).
3. Пуска отново гейта: RSpec, RuboCop и Brakeman докладват всички зелено, и този път отрицателният
   случай на AC5 реално е един от примерите, правещи RSpec зелен.

### Стъпка 5, отново — Review (ОДОБРЕНО)

**Същият фреш reviewer** проверява отново собствените си две констатации спрямо новия diff:
заявката вече е обхваната (scoped), отрицателният пример съществува и е потвърдено, че се е
провалял преди поправката (логът по-горе). Пуска отново пълния гейт независимо още веднъж.

**Присъда на reviewer-а:**
> **ОДОБРЕНО (APPROVE).** Двете констатации от предишния кръг са разрешени: `show` е обхванат
> (scoped) към `Current.user.orders`; отрицателният RSpec пример съществува, потвърдено е, че се е
> провалял срещу стария код, и минава срещу новия код. AC1–AC6 всички се свързват с минаващи примери.
> RSpec/RuboCop/Brakeman са чисти. Никой файл извън „Активи за произвеждане" на `FEATURE-2` не е
> променен.

### Стъпка 6 — Merge

**Архитектът прави merge.** Тялото на PR-а свързва всеки критерий за приемане с доказателството му,
и — за разлика от праволинейния успех на Част А — изрично отбелязва единия кръг на review, който
улови IDOR-а, защото секцията за процеса на `docs/definition-of-done.md` иска точно това: следата от
доказателства, не само крайната зелена отметка.

---

---

## Част В — FEATURE-3: каталог с продукти (два специализирани дефекта, уловени и поправени)

> Добавено от SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents. Същата конвенция като Части А и Б: реални
> имена на файлове, реална спецификация, реален commit-нат код, реални разсъждения от review; но
> илюстративни конзолни байтове за всяко изпълнение на RSpec/html-proofer/axe, защото тази среда
> (sandbox) няма нито Ruby toolchain, нито браузър (няма `chromedriver` на `PATH` тук — виж бележката
> за средата в [`03-rails-estore-sdlc.md`](../03-rails-estore-sdlc.md)).

### Стъпка 1 — Заявка → спецификация

**Prompt на собственика към архитекта:**
> „Регистрацията и checkout-ът работят. Сега направи реалните страници с продукти нещо, което
> търсеща машина може да индексира, и нещо, което потребител на екранен четец може да ползва — в
> момента дори няма layout файл."

**Архитектът** пише
[`docs/features/FEATURE-3-product-catalog-seo-a11y.md`](../code/rails-estore/docs/features/FEATURE-3-product-catalog-seo-a11y.md):
девет критерия за приемане (AC1–AC9), покриващи семантична/обозначена (labelled) markup, уникални
заглавия, Open Graph, Product + BreadcrumbList JSON-LD, `robots.txt`/sitemap, и гейта axe/html-proofer
— и, в термините на `docs/architecture.md`, назовава две НОВИ роли на reviewer, които тази
функционалност съществува, за да упражни: `seo-optimizer.md` и `frontend-qa.md`, диспечирани заедно с
общия `reviewer.md`, не вместо него.

### Стъпка 2 — Заземяване на неизвестните

И четирите твърдения, които спецификацията цитира, вече бяха заземени:
`NOTE-SDLC-4-ADD-1-gem-npm-versions.md` (петте gem-а + `@lhci/cli`, фиксирани версии),
`NOTE-SDLC-4-ADD-2-schema-product.md` (формата на Product JSON-LD),
`NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md` (проверките на WCAG, които axe налага),
`NOTE-SDLC-4-ADD-4-robots-sitemap-og.md` (`robots.txt`/sitemap/Open Graph). Не е нужно ново
диспечиране.

### Стъпка 3 — Първо провалящ се тест, после имплементация (два пропуска, по един във всяка специализирана област)

**Implementer, стъпка 1:** пише `spec/system/accessibility_spec.rb` и `spec/system/seo_spec.rb`
срещу stub изгледи, потвърждава, че и двата се провалят по правилната причина (все още няма layout,
няма meta тагове, няма JSON-LD).

**Implementer, стъпка 2:** пише реалния
[`ProductsController`](../code/rails-estore/app/controllers/products_controller.rb),
[`ProductsHelper`](../code/rails-estore/app/helpers/products_helper.rb) и layout-а. Излизат два
пропуска, по един във всяка нова специализирана област:

```erb
<%# app/views/products/index.html.erb — AS FIRST WRITTEN (the planted frontend-qa slip) %>
<%= form_with url: products_path, method: :get, role: "search" do |f| %>
  <div>
    <%= f.search_field :q, value: params[:q], placeholder: "Search products" %>
    <%= f.submit "Search" %>
  </div>
<% end %>
```

Липсва `<%= f.label :q, ... %>` — placeholder не е label (WCAG 1.3.1 / 4.1.2; собственият цитат на
`NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md`: „Placeholder текст не е достатъчен — той изчезва веднага
щом започнеш да пишеш и не се обявява надеждно от екранен четец"). И:

```erb
<%# app/views/products/show.html.erb — AS FIRST WRITTEN (the planted seo-optimizer slip) %>
<h1><%= @product.name %></h1>
<p><%= @product.description %></p>
<p>$<%= "%.2f" % @product.price_dollars %></p>
<%# ... no <script type="application/ld+json"> block at all %>
```

Никъде на страницата няма Product JSON-LD.

### Стъпка 4 — Гейт (зелен за това, което можа да пусне)

```text
[REFERENCE — illustrative]
$ bundle exec rspec spec/models spec/requests
..........................
26 examples, 0 failures

$ bundle exec rspec spec/system/accessibility_spec.rb spec/system/seo_spec.rb
[verify] no chromedriver on PATH in this environment — system specs requiring `js: true` cannot run
here; reproduce on a machine with Chrome + chromedriver installed (README.md, "Frontend gate").
2 examples, 0 failures, 2 not run
```

`bundle exec rubocop` и `bundle exec brakeman -q --no-summary` докладват и двата чисто — стилът
няма мнение за липсващ `<label>`, а проверките на Brakeman изобщо не покриват категориите WCAG или
SEO [източник: `NOTE-SDLC-4-3-brakeman-checks.md`, същата пролука „не е в неговия каталог", която
IDOR случаят в тази глава вече показа за авторизацията]. Implementer-ът докладва всеки гейт, който
реално успя да пусне, като зелен, и предава за review, честно вярвайки, че FEATURE-3 е готова.

### Стъпка 5 — Review (две ПРОМЕНИ ЗАЯВЕНИ, паралелно)

**Архитектът диспечира трима reviewer-и**: общия `reviewer.md` (съответствие/test-first/заземяване/
обхват — нищо за отбелязване тук, изобщо няма `Current.user`-scoped заявка или извикване на
`permit()` в този контролер), и — нови за тази функционалност — `seo-optimizer.md` и
`frontend-qa.md` паралелно. И двата реално пускат инструментите си, вместо да се доверят на
бележката „системните спекове не се пуснаха" по-горе.

**Присъда на `frontend-qa`:**
> **ПРОМЕНИ ЗАЯВЕНИ (CHANGES REQUESTED).**
> 1. `app/views/products/index.html.erb` — полето за търсене `<input type="search" name="q">` няма
>    свързан `<label>`, `aria-label` или `aria-labelledby`. Правило на axe `label`, влияние
>    **критично**, WCAG 1.3.1 / 4.1.2. Потребител на екранен четец чува „поле за редактиране, празно"
>    без указание какво да въведе. Поправка: добави `<%= f.label :q, "Search products" %>` преди
>    полето.
>
> Бележка за покритието: автоматичните проверки на axe улавят ~30–40% от реалните WCAG проблеми
> (`NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md`); прочетох и рендираната markup на ръка и не намерих нищо
> друго — това не е същото твърдение като „тази страница е напълно достъпна."

**Присъда на `seo-optimizer`:**
> **ПРОМЕНИ ЗАЯВЕНИ (CHANGES REQUESTED).**
> 1. `app/views/products/show.html.erb` — никъде на страницата няма блок `<script
>    type="application/ld+json">` за Product. Google Merchant Listing изисква минимум `name`,
>    `image`, `offers.price`, `offers.priceCurrency` (`NOTE-SDLC-4-ADD-2-schema-product.md`); без
>    него тази страница изобщо не е допустима за rich result от merchant listing, и AC5 не е
>    удовлетворен. Поправка: рендирай `product_json_ld(@product)` (вече дефинирана в
>    `ProductsHelper`, просто не е извикана от изгледа).

### Стъпка 3, отново — поправката

**Implementer-ът** добавя липсващото `<%= f.label :q, "Search products" %>` в `index.html.erb`, и
добавя двата реда `<script type="application/ld+json">` (Product + BreadcrumbList) в
`show.html.erb` — точно версиите, commit-нати под
[`app/views/products/`](../code/rails-estore/app/views/products/). Пуска отново гейта:

```text
[REFERENCE — illustrative, on a machine with chromedriver on PATH]
$ bundle exec rspec spec/system/accessibility_spec.rb spec/system/seo_spec.rb
....
4 examples, 0 failures

$ bundle exec rake html_proofer:check
Running ["ScriptCheck", "LinkCheck", "ImageCheck"] on ["tmp/html_proofer/index.html", "tmp/html_proofer/show.html"] ... 

HTML-Proofer finished successfully.
```

### Стъпка 5, отново — Review (ОДОБРЕНО, и двата специалиста + общият reviewer)

`frontend-qa` пуска отново `accessibility_spec.rb` и потвърждава, че поправката с label-а разрешава
находката от axe. `seo-optimizer` парсва отново рендирания JSON-LD и потвърждава, че `name`/`image`/
`offers.price`/`offers.priceCurrency` всички са налице и валидни, плюс препоръчаните
`sku`/`brand`/`availability`. Общият `reviewer.md` потвърждава отново съответствието и обхвата. И
трите: **ОДОБРЕНО (APPROVE)**.

### Стъпка 6 — Merge

**Архитектът прави merge.** Тялото на PR-а свързва AC1–AC9 с доказателството им, и назовава
одобренията и на двата специалиста заедно с това на общия reviewer — секцията SEO & Достъпност на
`docs/definition-of-done.md` изисква и двете, не само едно от двете.

---

## Какво всъщност показват тези три транскрипта

Част А е как изглежда „цикълът проработи": спецификация → заземяване → test-first → зелен гейт →
чист review → merge, без изненади. Част Б е уроци по сигурност: агент без злонамерено намерение,
следвайки спецификацията, пишейки реални тестове, и все пак пуска реална уязвимост в продукция,
защото три автоматизирани гейта имаха законна причина да останат зелени — стъпката на фреш
reviewer-а за проверка на авторизацията на ръка, умишлена инструкция в
[`.claude/agents/reviewer.md`](../code/rails-estore/.claude/agents/reviewer.md), е това, което я
улови. Част В прави същата точка за *различна* двойка измерения на качеството: RuboCop и Brakeman
изобщо нямат SEO или accessibility проверки в каталога си — не пролука в иначе широк скенер, а
категория, която никога не са били построени да покриват — така липсващ `<label>` и липсващ Product
JSON-LD блок минават през всеки гейт, който този проект имаше *преди* този addendum.
`seo-optimizer.md` и `frontend-qa.md` са това, което затваря тази пролука, по същия начин, по който
фреш reviewer затвори пролуката в авторизацията в Част Б: назован специалист, с чеклист, инструктиран
да провери на ръка това, което никой наличен инструмент не проверява автоматично. Това е целият
аргумент на главата, доказан вече на три различни класа дефекти: управлението (governance) е това,
което прави AI-подпомогнатата разработка безопасна, не добрите намерения на модела, и дори не само
добър автоматизиран гейт.
