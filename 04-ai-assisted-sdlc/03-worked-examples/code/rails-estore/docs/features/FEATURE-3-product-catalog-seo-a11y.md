# FEATURE-3: Каталог с продукти — SEO-завършени, достъпни страници на витрината

**Status:** approved
**Owner:** архитект (Opus)
**Routing:** implementer=Sonnet · research=NOTE-SDLC-4-ADD-1, NOTE-SDLC-4-ADD-2, NOTE-SDLC-4-ADD-3,
NOTE-SDLC-4-ADD-4 · review=Sonnet (fresh) + `seo-optimizer` (Sonnet) + `frontend-qa` (Sonnet)

## Намерение (Intent)

FEATURE-1 и FEATURE-2 дадоха на този магазин система за акаунти и checkout — но самите страници с
продукти, с които купувач (или търсеща машина, или линк за споделяне в социална мрежа) се среща
първо, никога не бяха SEO- или accessibility-завършени: нямаше `<title>`/description на страница,
нямаше структурирани данни, нямаше `robots.txt`/sitemap, и — до тази функционалност — изобщо нямаше
`app/views/layouts/application.html.erb`, така че нямаше къде да живее `<html lang>`, viewport meta
tag, или skip link. Тази функционалност затваря тази пролука и е носителят на двама нови
специализирани ревюиращи, които този addendum добавя към състава: **`seo-optimizer`**
(откриваемост) и **`frontend-qa`** (достъпност/качество на frontend-а) — нито един не се припокрива
с фокуса на `reviewer.md` върху авторизация/mass assignment, и нито един не се припокрива с
RuboCop/Brakeman, които нямат мнение по липсващ `<label>` или липсващ Product JSON-LD блок.

## Критерии за приемане (Acceptance criteria)

- AC1 — `GET /products` (index) рендира точно един `<h1>`, форма за търсене, чието текстово поле има
  програмно обвързан `<label>` (не само placeholder), и — за всеки продукт с `image_url` — `<img>` с
  описателен `alt` текст.
- AC2 — `GET /products/:id` (show) рендира точно един `<h1>` и, когато продуктът има `image_url`,
  `<img>` с описателен `alt` текст.
- AC3 — всяка страница (`index`, `show`) задава уникален `<title>` и meta description чрез
  `set_meta_tags`/`display_meta_tags` (gem-ът `meta-tags`) — title-ът на `index` и този на `show`
  никога не са идентични — и `<link rel="canonical">`, сочещ към собствения URL на тази страница
  [source: NOTE-SDLC-4-ADD-1-gem-npm-versions.md, NOTE-SDLC-4-ADD-4-robots-sitemap-og.md].
- AC4 — всяка страница носи четирите задължителни Open Graph свойства (`og:title`, `og:type`,
  `og:image`, `og:url`) с абсолютен `og:image` URL
  [source: NOTE-SDLC-4-ADD-4-robots-sitemap-og.md].
- AC5 — product show страницата носи валиден schema.org **Product** JSON-LD с, като минимум, `name`,
  `image`, `offers.price` (числов низ > 0), `offers.priceCurrency` ("USD"); `sku`, `brand`, и
  `offers.availability` (`https://schema.org/InStock`) присъстващи като препоръчаните полета, плюс
  **BreadcrumbList** JSON-LD, назоваващ Home → Products → продукта
  [source: NOTE-SDLC-4-ADD-2-schema-product.md].
- AC6 — `public/robots.txt` съществува, препраща към sitemap-а, и не прави `Disallow` на `/products`;
  `config/sitemap.rb` (`sitemap_generator`) изброява products index-а и show URL на всеки продукт
  [source: NOTE-SDLC-4-ADD-4-robots-sitemap-og.md].
- AC7 — `expect(page).to be_axe_clean.according_to(:wcag21aa)` от
  `spec/system/accessibility_spec.rb` минава както на index, така и на show страницата — нула
  автоматизирани нарушения на WCAG 2.1 AA
  [source: NOTE-SDLC-4-ADD-1-gem-npm-versions.md, NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md].
- AC8 — `bundle exec rake html_proofer:check` не докладва счупени вътрешни връзки и без липсващи
  атрибути `alt` върху рендирания index/show HTML
  [source: NOTE-SDLC-4-ADD-1-gem-npm-versions.md].
- AC9 — `app/views/layouts/application.html.erb` задава `<html lang="en">`, включва skip-to-content
  връзка, която става видима при фокус с клавиатура, и структурира страницата с landmark-и
  `<header>`/`<nav>`/`<main>`/`<footer>` [source: NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md].

## Твърдения за заземяване (Claims to ground)

- Версии и извикване на `meta-tags`/`sitemap_generator`/`html-proofer`/`axe-core-rspec`/
  `axe-core-capybara`/`@lhci/cli` — заземено, `docs/research/NOTE-SDLC-4-ADD-1-gem-npm-versions.md`.
- Форма на schema.org Product JSON-LD + задължителни/препоръчани полета на Google Merchant Listing —
  заземено, `docs/research/NOTE-SDLC-4-ADD-2-schema-product.md`.
- WCAG/axe проверки (етикети, alt, контраст, ред на заглавията, lang, фокус, landmark-и) —
  заземено, `docs/research/NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md`.
- Конвенции за `robots.txt`/`sitemap.xml` и задължителни свойства на Open Graph — заземено,
  `docs/research/NOTE-SDLC-4-ADD-4-robots-sitemap-og.md`.

## Извън обхват (Out of scope)

- Страници за категории/колекции и по-дълбоката верига от breadcrumb-и, от която биха се нуждали
  (този каталог е единичен, плосък index `/products`; BreadcrumbList, който AC5 иска, е винаги точно
  три нива дълбоко).
- Схема `AggregateRating`/`Review` — все още не съществува функционалност за отзиви, от която да се
  черпят.
- Тематична система от цветове — тази функционалност разчита на стандартния черно-на-бяло текст на
  браузъра, който вече покрива минимума за контраст 4.5:1; преминаване към дизайн система с
  персонализирани цветове е отделна функционалност, която би се нуждаела от собствен одит на
  контраста.
- Реално пускане на `npx lhci autorun` в собствения CI на този проект — окабелен е като документиран,
  заземен **референтен** гейт (`.lighthouserc.json`, `package.json`), който читателят възпроизвежда
  на машина с Node, съгласно обхвата на addendum-а; `.claude/hooks/verify.sh` не го извиква.

## Активи за изготвяне (Assets to produce)

- `.claude/agents/seo-optimizer.md`, `.claude/agents/frontend-qa.md`
- `app/controllers/products_controller.rb` (разширен: `set_meta_tags`, търсене)
- `app/views/products/index.html.erb`, `app/views/products/show.html.erb` (разширени)
- `app/views/layouts/application.html.erb` (нов)
- `app/helpers/products_helper.rb` (нов — `product_json_ld`, `breadcrumb_json_ld`)
- `app/models/product.rb` (разширен: `image_url`, `sku`, `price_currency`)
- `db/schema.rb` (разширен: `products.image_url`, `products.sku`)
- `db/seeds.rb`, `config/database.yml` (нови)
- `public/robots.txt`, `config/sitemap.rb` (нови)
- `spec/system/accessibility_spec.rb`, `spec/system/seo_spec.rb`, `spec/support/capybara_driver.rb`
  (нови)
- `lib/tasks/html_proofer.rake` (нов)
- `.lighthouserc.json`, `package.json` (нови — конфигурацията на референтния Node гейт)
- `Gemfile`, `.claude/hooks/verify.sh`, `CLAUDE.md`, `docs/definition-of-done.md` (окабеляване)

## Гейтове (Gates)

Вход: тази спецификация одобрена; FEATURE-1 + FEATURE-2 мержнати (страниците на каталога
преизползват `Product` и `authenticated?` от тези функционалности); четирите бележки за заземяване
`NOTE-SDLC-4-ADD-*` пристигнали. Изход: чеклистът `docs/definition-of-done.md` изцяло, включително
новата секция **SEO & Достъпност** — с одобрение от `seo-optimizer.md` И `frontend-qa.md`, в
допълнение към одобрението на общия `reviewer.md`.
