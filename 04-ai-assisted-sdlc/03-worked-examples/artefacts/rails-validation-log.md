# Валидационен лог — скелет на rails-estore/

Реален, заснет изход от валидирането на всеки governance файл под
[`code/rails-estore/`](../code/rails-estore/), плюс реален secret-scan през цялото дърво, пуснат в
собствения sandbox на това хранилище (Windows 11, Python 3.13, Git Bash) по време на писането на
тази глава — същия sandbox и същия метод, използвани от
[`validation-log.md`](../../02-local-environment-setup/../03-worked-examples/artefacts/validation-log.md)
на Java главата. **§§1–3, §5 и §7 по-долу са реални, заснети доказателства** — всяка команда реално
беше изпълнена. **§4 и §6 са илюстративни** — това хранилище няма Ruby/Rails toolchain (`bundle`,
`rspec`, `rubocop`, `brakeman` не са инсталирани тук), нито браузър/chromedriver/Node, затова изходът
на RSpec/RuboCop/Brakeman (§4) и на axe/html-proofer/Lighthouse CI (§6, добавен от
SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents) е представен като заземена референция (точна команда,
точни фиксирани версии от `research/NOTE-SDLC-4-1-versions.md` и
`research/NOTE-SDLC-4-ADD-1-gem-npm-versions.md`), не заснето изпълнение — виж бележката за средата
в [`03-rails-estore-sdlc.md`](../03-rails-estore-sdlc.md) защо. **§7 е единственото изключение от
цялата тази история**: самият Docker Е наличен в този sandbox
(SPEC-SDLC-4-ADDENDUM-2-docker-compose), затова пътят с docker-compose реално беше build-нат, вдигнат,
curl-нат и тестван от край до край — реален изход, реални грешки, открити и поправени, реален teardown.

> Бележка за формата: изходът на инструментите (JSON, `OK ...` редовете на
> `validate_frontmatter.py`, RSpec/RuboCop/Brakeman, grep, axe/html-proofer/Lighthouse, Docker/curl/
> rspec) остава на собствения си (английски) език на инструмента — превежда се само прозата около
> него. §3 по-долу е изключение: тъй като hook скриптовете `context.sh`/`guard.sh`/`verify.sh` вече
> са преведени на български, реалният им отпечатан изход вече Е на български, и е заснет наново
> по-долу.

## 1. `settings.json` се парсва като валиден JSON

```
$ .venv/Scripts/python.exe -m json.tool code/rails-estore/.claude/settings.json
{
    "$schema": "https://json.schemastore.org/claude-code-settings.json",
    "hooks": {
        "PostToolUse": [
            {
                "matcher": "Edit|Write",
                "hooks": [
                    { "type": "command", "command": ".claude/hooks/verify.sh" }
                ]
            }
        ],
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    { "type": "command", "command": ".claude/hooks/guard.sh" }
                ]
            }
        ],
        "SessionStart": [
            {
                "hooks": [
                    { "type": "command", "command": ".claude/hooks/context.sh" }
                ]
            }
        ]
    }
}
```

Същата форма с три събития, `{matcher, hooks:[{type:"command", command:...}]}`, както в собствения
`.claude/settings.json` на `ds_ml_ai_starter` и порта на `java-project` — схемата
(`json.schemastore.org/claude-code-settings.json`) нито знае, нито я интересува на какъв език
проверяват hook-овете. `python -m json.tool` излиза с ненулев код при повреден JSON; чист,
форматиран echo, както по-горе, е сигналът за успех.

## 2. Frontmatter-ът на агентите е добре формиран YAML и отговаря на документираната схема

`code/validate_frontmatter.py` (парсва разделения с `---` frontmatter с PyYAML, проверява дали
`name` + `description` присъстват, дали `model` е една от документираните стойности, дали всеки
друг ключ е в документирания набор от опционални полета) пуснат срещу всичките пет файла от състава:

```
$ .venv/Scripts/python.exe code/validate_frontmatter.py \
    code/rails-estore/.claude/agents/researcher.md \
    code/rails-estore/.claude/agents/implementer.md \
    code/rails-estore/.claude/agents/reviewer.md \
    code/rails-estore/.claude/agents/seo-optimizer.md \
    code/rails-estore/.claude/agents/frontend-qa.md
OK   code/rails-estore/.claude/agents/researcher.md: name='researcher' model='haiku' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/implementer.md: name='implementer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/reviewer.md: name='reviewer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/seo-optimizer.md: name='seo-optimizer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   code/rails-estore/.claude/agents/frontend-qa.md: name='frontend-qa' model='sonnet' fields=['description', 'model', 'name', 'tools']
```

Моделното рутиране съответства на `docs/architecture.md` §2 (researcher на Haiku, implementer/
reviewer на Sonnet) плюс двамата специалисти, добавени от този addendum (`seo-optimizer`,
`frontend-qa`, и двата Sonnet) — виж секцията Model routing на `CLAUDE.md`.

## 3. Hook-овете реално се изпълняват и налагат това, което твърдят

`context.sh` (`SessionStart`), пуснат директно (реален изход, заснет наново на 2026-09-05 — вече на
български, защото самият hook скрипт беше преведен):

```
$ bash .claude/hooks/context.sh
──────────────────────────────────────────────────────────
 rails-estore — управляван SDLC скелет за Rails онлайн магазин
──────────────────────────────────────────────────────────
 Прочети първо: docs/architecture.md · docs/definition-of-done.md · CLAUDE.md
 Роли: Opus (архитект, главна сесия) · .claude/agents/researcher.md (Haiku) ·
        .claude/agents/implementer.md (Sonnet) · .claude/agents/reviewer.md (свеж Sonnet)
        За UI функционалности също: .claude/agents/seo-optimizer.md · .claude/agents/frontend-qa.md (Sonnet)

 Feature spec-ове:
   • FEATURE-1: Регистрация, вход и изход на потребител  approved
   • FEATURE-2: От количка до поръчка — checkout с фиктивен (stubbed) платежен интерфейс  approved
   • FEATURE-3: Каталог с продукти — SEO-завършени, достъпни страници на витрината  approved

 Напомняния за гейта: провалящ се RSpec пример commit-нат преди кода, който покрива ·
 bundle exec rspec минава · rubocop чист · brakeman -q --no-summary чист ·
 никъде няма жив Stripe ключ (sk_live_/pk_live_) · независим review преди merge.
 (docs/definition-of-done.md)
──────────────────────────────────────────────────────────
exit=0
```

(Логът по-горе изброява и трите feature spec-а — FEATURE-1/2/3 — понеже и трите вече са merge-нати в
това хранилище към момента на повторното изпълнение; по-ранната версия на този лог показваше само
първите две.)

`guard.sh` (`PreToolUse`), захранен със синтетични hook payload-и на stdin точно както би го направил
Claude Code — пет случая, реален изход, заснет наново:

```
$ printf '{"command":"bundle exec rspec"}' | bash guard.sh
exit=0

$ printf '{"command":"rm -rf /"}' | bash guard.sh
[guard] BLOCKED: rm -rf /
exit=2

$ printf '{"command":"git commit --no-verify -m test"}' | bash guard.sh
[guard] BLOCKED: git commit --no-verify (пропуска pre-commit гейта на RSpec/RuboCop/Brakeman)
exit=2

$ printf '{"command":"curl -H XSomeKey:sk_live_EXAMPLE_redacted_not_a_real_key https://example.com"}' | bash guard.sh
[guard] BLOCKED: в команда се появи шаблон на жив Stripe ключ (sk_live_/pk_live_/rk_live_) — използвай само pk_test_/sk_test_, от .env, никога inline
exit=2

$ printf '{"command":"echo STRIPE_SECRET_KEY=$STRIPE_SECRET_KEY"}' | bash guard.sh
[guard] BLOCKED: отпечатване на тайна в stdout
exit=2
```

Безобидна build команда минава напълно (exit 0); деструктивна файлова команда, commit, заобикалящ
hook-овете, шаблон на **жив** Stripe ключ, и отпечатване на секретна env променлива са всички
блокирани (exit 2). Забележи какво НЕ задейства правилото за жив ключ: всеки плейсхолдър
`pk_test_`/`sk_test_`, използван навсякъде в кода и документацията на този проект, остава недокоснат
— регулярният израз засича само инфикса `_live_`.

`verify.sh` (`PostToolUse`), същият подход със синтетичен payload, реален изход, заснет наново:

```
$ printf '{"file_path":"app/controllers/checkout/orders_controller.rb"}' | bash verify.sh
[verify] bundle не е на PATH — прескача се гейтът за app/controllers/checkout/orders_controller.rb
exit=0

$ printf '{"file_path":"README.md"}' | bash verify.sh
[verify] няма проверки за: README.md
exit=0

$ printf '{"file_path":"Gemfile"}' | bash verify.sh
[verify] bundle не е на PATH — прескача се проверката
exit=0
```

`verify.sh` деградира елегантно, когато `bundle` липсва (този sandbox), вместо да провали целия
hook — същият договор „прескочи елегантно, когато инструмент липсва", който
`java-project/.claude/hooks/verify.sh` документира за липсващ `mvn`. (Потвърдено с
`command -v bundle`, който излиза с код 1 в тази среда — `bundle` наистина липсва, съобщенията по-горе
не са симулация.)

## 4. RSpec / RuboCop / Brakeman — референтна репродукция (не е изпълнено в тази среда)

Тук няма инсталиран Ruby/Rails toolchain (`which bundle`, `which rspec` не намират нищо). Командите,
имената на файловете и очакваните резултати по-долу са заземени в
`research/NOTE-SDLC-4-1-versions.md` (фиксирани версии на gem-ове) и
`research/NOTE-SDLC-4-3-brakeman-checks.md` (реалните категории проверки и форма на изхода на
Brakeman); възпроизведи реално на машина с инсталирани Ruby 4.0.6 + Rails 8.1.3.1, отвътре в
`code/rails-estore/` след `bundle install`. Изходът на инструментите по-долу остава на английски —
това е реалният език, на който тези инструменти отпечатват.

```text
[REFERENCE — illustrative RSpec output, gem versions per NOTE-SDLC-4-1-versions.md]
$ bundle exec rspec
.........................

Finished in 1.84 seconds (files took 2.11 seconds to load)
25 examples, 0 failures

Randomized with seed 48213
```

```text
[REFERENCE — illustrative RuboCop output]
$ bundle exec rubocop
Inspecting 24 files
........................

24 files inspected, no offenses detected
```

```text
[REFERENCE — illustrative Brakeman output shape, per NOTE-SDLC-4-3-brakeman-checks.md's documented
 output format — text/JSON/11 formats total]
$ bundle exec brakeman -q --no-summary

== Brakeman Report ==

No warnings found
```

### Предупреждение от Brakeman, уловено и разрешено (случаят, за който Brakeman Е построен)

В началото на разработката на FEATURE-2, `app/views/products/show.html.erb` имаше един ред, различен
от версията, commit-ната под `code/rails-estore/`:

```erb
<%# EARLY DRAFT — flagged and then fixed, not what's committed %>
<p><%= @product.description.html_safe %></p>
```

`.html_safe` казва на Rails „не HTML-escape-вай този string" — правилно за съдържание, което самото
приложение изцяло контролира, грешно за каквото и да е, произхождащо от потребителски или admin
вход, защото превръща съхранен string директно в суров HTML, който браузърът изпълнява.
`docs/research/NOTE-SDLC-4-3-brakeman-checks.md` назовава точно този шаблон като една от основните
проверки на Brakeman:

```text
[REFERENCE — illustrative, matching Brakeman's documented "Cross Site Scripting" warning shape]
$ bundle exec brakeman -q --no-summary

== Brakeman Report ==

+ Confidence: High
  Category: Cross Site Scripting
  Check: CrossSiteScripting
  Message: Unescaped model attribute
  File: app/views/products/show.html.erb
  Line: 3
  Code: @product.description.html_safe

1 warning found
```

Стъпката `brakeman -q --no-summary` на `.claude/hooks/verify.sh` се проваля с ненулев код при това —
поправката (премахни `.html_safe`; ERB HTML-escape-ва по подразбиране, точно това прави commit-натия
[`app/views/products/show.html.erb`](../code/rails-estore/app/views/products/show.html.erb)) връща
Brakeman на нула предупреждения, преди implementer-ът да докладва гейта зелен. Съпостави това с
находката IDOR от транскрипта на checkout цикъла, която Brakeman **не** улавя — двете заедно са
честната картина: Brakeman е реален и полезен за класовете уязвимости, които цели (SQLi, mass
assignment, XSS), а фреш, насочван от човек reviewer все още е нужен за класовете, които не покрива
(счупен контрол на достъпа между записи от същия тип).

## 5. Secret scan — реален, заснет, чист

Изискано, преди тази глава да може да отчете завършеност: grep-ване на цялото дърво `rails-estore/`
за нещо, наподобяващо реален ключ. Заснето наново на 2026-09-05 (изходът по-долу е смесен
български/английски, защото по-голямата част от `rails-estore/` — hook-овете, docs, README — вече е
преведена на български, докато самите ключови шаблони `sk_live_`/`pk_live_`/`rk_live_` умишлено
остават на английски навсякъде):

```
$ grep -rniE 'sk_live|pk_live|rk_live' --exclude-dir=tmp code/rails-estore/
code/rails-estore/.claude/agents/reviewer.md:42:7. **Тайни:** grep-ни диффа за нещо, наподобяващо реален ключ (`sk_live`, `pk_live`, гол 40-символен
code/rails-estore/.claude/hooks/context.sh:29:echo " никъде няма жив Stripe ключ (sk_live_/pk_live_) · независим review преди merge."
code/rails-estore/.claude/hooks/guard.sh:27:  deny "в команда се появи шаблон на жив Stripe ключ (sk_live_/pk_live_/rk_live_) — използвай само pk_test_/sk_test_, от .env, никога inline"
code/rails-estore/.env.example:2:# TEST-MODE values only. NEVER put a live key (sk_live_/pk_live_/rk_live_) here or in any file that
code/rails-estore/docs/architecture.md:88:- **Без тайни:** нищо, наподобяващо жив ключ (`sk_live_`/`pk_live_`/`rk_live_`), никъде в диффа.
code/rails-estore/docs/definition-of-done.md:36:- [ ] **Без тайни:** нищо, наподобяващо жив ключ (`sk_live_`/`pk_live_`/`rk_live_`), никъде в диффа;
code/rails-estore/README.md:364:> --no-verify`, наглед жив Stripe ключ (`sk_live_`/`pk_live_`/`rk_live_`), `rm -rf /`, или принудителен

$ grep -rnE '[sp]k_(live|test)_[A-Za-z0-9]{10,}' --exclude-dir=tmp code/rails-estore/
(no matches)
```

(`--exclude-dir=tmp` изключва `code/rails-estore/tmp/`, локален, gitignore-нат bootsnap кеш, останал
от секция 7's Docker изпълнение по-долу — не част от commit-натото дърво, и не носи никакво
съдържание за прозата.)

Всяко попадение на `sk_live`/`pk_live`/`rk_live` от първия grep е проза, *описваща* правилото (в hook,
документ, или собствения чеклист на reviewer-а) — нито едно не е реална стойност на ключ. Вторият
grep, който търси нещо с формата на реален ключ (префикс `pk_/sk_` последван от дълга поредица
ключови символи), връща **нищо**: `.env.example` използва изписани с думи плейсхолдъри
(`sk_test_your_secret_key_here`), не низ с формата на ключ, така че няма за какво скенер да се
хване. **Чисто.**

> **Бележка от реалния свят.** По-ранна чернова на точно този файл използваше `sk_test_`, последван
> от 24 буквални `X`-а като плейсхолдър. Собствената **push protection** на GitHub отхвърли push-а
> въпреки това — неговият Stripe детектор засича по *форма*, а `sk_test_` + 24 символа е точно тази
> форма, X-ове или не. Поправката беше да се използва изписан с думи плейсхолдър без низ с дължина на
> ключ. Това е същата защита на дълбочина (defence-in-depth), която тази глава проповядва, уловена
> един слой по-навън от собствения ни `guard.sh` — точно това е идеята.

## 6. Frontend гейт — референтен изход на axe / html-proofer / Lighthouse CI (добавен от този addendum)

Същата конвенция като §4: в тази среда (sandbox) няма нито браузър (няма `chromedriver`), нито
Node/`npx`, затова всеки блок по-долу е **заземена референтна репродукция**, не заснето изпълнение —
точни команди и точни фиксирани версии от `docs/research/NOTE-SDLC-4-ADD-1-gem-npm-versions.md`.
Възпроизведи реално на машина с инсталирани Ruby 4.0.6 + Chrome/chromedriver + Node 24, отвътре в
`code/rails-estore/` след `bundle install` и `npm install` (виж `README.md`, „Frontend gate").
Изходът на инструментите по-долу остава на английски.

**axe, улавящ внедрения a11y пропуск на FEATURE-3** (преди поправката в
[`app/views/products/index.html.erb`](../code/rails-estore/app/views/products/index.html.erb) — виж
транскрипта на цикъла, Част В):

```text
[REFERENCE — illustrative axe violation shape, per NOTE-SDLC-4-ADD-3-wcag-a11y-checks.md's documented
 axe output fields (id, impact, description, help, helpUrl, nodes, tags)]
$ bundle exec rspec spec/system/accessibility_spec.rb
F.

Failures:

  1) Product catalog accessibility has no automatically detectable WCAG 2.1 AA violations on the catalog index
     Failure/Error: expect(page).to be_axe_clean.according_to(:wcag21aa)
       expected no accessibility violations, but got 1:

       Violation: label (critical)
         Form elements must have labels
         https://dequeuniversity.com/rules/axe/4.13/label
         Node: input[name="q"]
         Tags: wcag2a, wcag412, wcag131, section508

2 examples, 1 failure
```

**axe, след поправката:**

```text
[REFERENCE — illustrative]
$ bundle exec rspec spec/system/accessibility_spec.rb
..

2 examples, 0 failures
```

**html-proofer**, пуснат чрез `lib/tasks/html_proofer.rake` (5.2.2 — програмно
`HTMLProofer.check_directory(...).run`, точно цитираната употреба в
`NOTE-SDLC-4-ADD-1-gem-npm-versions.md`):

```text
[REFERENCE — illustrative]
$ bundle exec rake html_proofer:check
Running ["ScriptCheck", "LinkCheck", "ImageCheck"] on ["tmp/html_proofer/index.html", "tmp/html_proofer/show.html"] ...

HTML-Proofer finished successfully.
```

**Lighthouse CI** (`@lhci/cli` 0.15.1, npm — само референтен Node гейт; не се извиква от
`.claude/hooks/verify.sh`, който е скрипт само за Ruby):

```text
[REFERENCE — illustrative Lighthouse CI summary shape]
$ npm install
$ npx lhci autorun
✅  1 result(s) for http://localhost:3000/products
Accessibility:    98/100
SEO:              100/100
Best Practices:   96/100
Performance:      89/100

No LHCI budget assertions failed.
```

Съпостави тази секция с примера на Brakeman от §4 по същия начин, по който сдвояването на Част Б/
Част В в главата го прави: автоматизираните инструменти са реални и всеки улавя реален клас дефект
(XSS чрез `.html_safe` за Brakeman; липсващ `<label>` за axe) — и всеки има граница, очертана от
това, което изобщо проверява, което е точно причината `seo-optimizer.md` и `frontend-qa.md` да
съществуват като назовани, ръчни стъпки на review, вместо „изчакай инструментът да го улови."

## 7. Docker Compose — реален, заснет, от край до край (SPEC-SDLC-4-ADDENDUM-2)

**Тази секция е изцяло реално изпълнена**, за разлика от §4/§6 по-горе — Docker 28.4.0 / Docker
Compose v2.39.2 реално са налични в тази среда (sandbox), затова важи собственото правило на
addendum-а („ако можеш да го пуснеш тук, пусни го реално"). Хост: Windows 11, Docker Desktop, отвътре
в `code/rails-estore/`.

### `docker compose build` — първият опит се провали, реална грешка, реална поправка

```text
$ docker compose build
...
[5/7] RUN bundle install:
Resolving dependencies...
Could not find compatible versions
Because rubocop-rails >= 2.37.0 depends on rubocop >= 1.89.0, < 2.0
  and Gemfile depends on rubocop = 1.86.0,
  rubocop-rails >= 2.37.0 cannot be used.
So, because Gemfile depends on rubocop-rails = 2.37.0,
  version solving has failed.
ERROR: process "/bin/sh -c bundle install" did not complete successfully: exit code: 6
```

`research/NOTE-SDLC-4-1-versions.md` фиксира `rubocop 1.86.0`; реалният `bundle install` —
разрешаващ зависимостите спрямо реални, актуални метаданни на rubygems.org, не спрямо бележка —
установи, че `rubocop-rails 2.37.0` вече изисква `rubocop >= 1.89.0`. Поправено чрез разхлабване на
фиксацията в Gemfile до `gem "rubocop", ">= 1.89.0", "< 2.0"`; Bundler разреши `rubocop 1.90.0`.
Rebuild-ът успя:

```text
$ docker compose build
...
Installing rubocop 1.90.0
...
Bundle complete! 20 Gemfile dependencies, 125 gems now installed.
...
 rails-estore-web  Built
```

### `docker compose up` — реален boot лог, `/up` връща 200

```text
$ docker compose up -d
$ docker logs rails-estore-web-1
Seeded 4 product(s).
Seeded 4 product(s).
=> Booting Puma
=> Rails 8.1.3.1 application starting in development
Puma starting in single mode...
* Puma version: 8.0.2 ("Into the Arena")
* Ruby version: ruby 4.0.6 (2026-07-14 revision 03b6d3f889) +PRISM [x86_64-linux]
*  Min threads: 3
*  Max threads: 3
*  Environment: development
*          PID: 1
* Listening on http://0.0.0.0:3000
Use Ctrl-C to stop
```

(„Seeded 4 product(s)." се появява два пъти: `db:prepare` на Rails 8 seed-ва автоматично първия път,
когато създава свежа база данни, а после `bin/docker-entrypoint` пуска изричния `db:seed`, който
спецификацията на addendum-а изисква — `find_or_create_by!` в `db/seeds.rb` прави втория пас no-op
отвъд реда в лога.)

```text
$ curl -s -w "\nHTTP_STATUS:%{http_code}\n" http://localhost:3000/up
<!DOCTYPE html><html><body style="background-color: green"></body></html>
HTTP_STATUS:200

$ curl -s -o /dev/null -w "HTTP_STATUS:%{http_code}\n" http://localhost:3000/
HTTP_STATUS:200

$ curl -s -o /dev/null -w "HTTP_STATUS:%{http_code}\n" http://localhost:3000/session/new
HTTP_STATUS:200

$ curl -s -o /dev/null -w "HTTP_STATUS:%{http_code}\n" http://localhost:3000/products/1
HTTP_STATUS:200
```

Страницата с каталога също беше отворена в реален браузър и screenshot-ната срещу работещия
контейнер — четирите seed-нати продукта (Rails Mug, Convention Over Configuration T-Shirt, Omakase
Sticker Pack, Migration Notebook) се рендират с имена, цени и работеща кутия за търсене; страницата
за вход рендира реална `<form>` с CSRF `authenticity_token`, обозначено (labelled) поле за имейл и
бутон за изпращане. Същата визуална проверка улови реална, вече съществуваща грешка в рендирането
(по-долу).

**Реална грешка, уловена от screenshot, не от четене на кода.**
`app/views/products/index.html.erb` имаше многоредов ERB коментар `<%# ... %>`, който сам по себе си
вграждаше буквален таг `<%= f.label %>` като описателен текст. ERB коментар се затваря на *първото*
срещнато `%>`, не на последното — затова се затвори на `%>`-то на вградения таг, и всеки символ от
коментара, написан *след* него (цяло изречение), изтече на рендираната страница като обикновен,
видим текст, точно под заглавието „Shop all products". Нищо в RSpec suite-а не улавя това (никой от
request специфициите (specs) не проверява за точно този низ), и не беше видимо при небрежно четене
на `.erb` изходния код — показа се едва когато страницата беше реално рендирана и разгледана.
Поправено чрез преписване на коментара без никакви вградени ERB тагове.

### `docker compose run --rm web bundle exec rspec` — реален pass/fail, итерирано до зелено

Първото реално изпълнение изкара наяве три истински грешки при boot, никоя от които не е app логика:

1. **Счупване на API-то на `rspec-rails`.** `config.fixture_path = ...` (в единствено число) на
   `spec/rails_helper.rb` е pre-8.0 API на rspec-rails; 8.0.4 (фиксираната версия на този проект)
   дефинира само множественото `fixture_paths=` — потвърдено чрез директно четене на
   `rspec-rails-8.0.4/lib/rspec/rails/configuration.rb` вътре в контейнера. Поправено:
   `config.fixture_paths = [Rails.root.join("spec/fixtures")]`.
2. **`RAILS_ENV` тихо грешен.** Docker образът задава `RAILS_ENV=development` по подразбиране (за да
   не изисква `docker compose up` никаква конфигурация); `ENV["RAILS_ENV"] ||= "test"` на
   `spec/rails_helper.rb` затова никога не се задейства, и *целият suite се пуска срещу
   `development.rb`*, не `test.rb`. Всеки request spec се проваляше с реална 403 страница „Blocked
   hosts: www.example.com" — allow-list-ът на хостове на `development.rb`, не логиката на самото
   приложение. Поправено: `ENV["RAILS_ENV"] = "test"` (твърдо присвояване) в `rails_helper.rb`.
3. **Две липсващи стойности по подразбиране на `test.rb`.** Веднъж щом RAILS_ENV наистина беше
   „test", две настройки, които собственият генериран `test.rb.tt` на Rails доставя (потвърдено чрез
   директно четене на файла-шаблон на `railties-8.1.3.1`), липсваха от този ръчно писан `test.rb`:
   `config.hosts << "www.example.com"` (подразбиращото се Host заглавие на integration-test сесията)
   и `config.action_controller.allow_forgery_protection = false` (иначе всеки request spec с
   `post`/`patch`/`delete` се проваля с `ActionController::InvalidAuthenticityToken`).
4. **Автоматичното require на `html-proofer` издърпа липсваща native библиотека.** Записът за
   `html-proofer` в `Gemfile` (`group :test`) нямаше `require: false`; `Bundler.require` затова
   автоматично го зареждаше при *всяко* boot-ване в test среда, транзитивно изисквайки
   `typhoeus -> ethon -> libcurl.so` — библиотека, която този минимален, без-Node образ не
   инсталира (нужна е само за собствените проверки на external-link на html-proofer, а
   `lib/tasks/html_proofer.rake` вече прави `require "html-proofer"` сам, точно преди употреба).
   Поправено: `require: false` на реда в Gemfile.

След тези четири поправки остана едно реално несъответствие на assertion — не грешка, а
взаимодействие между конфигурация и тестов стил: `config.action_dispatch.show_exceptions =
:rescuable` (също копирано дословно от собствения `test.rb.tt` на Rails) означава, че *rescuable*
грешка като `ActiveRecord::RecordNotFound` вече се улавя от routing-а на Rails и се превръща в 404
отговор, вместо да се разпространява като хвърлено изключение — така IDOR тестът на
`spec/requests/checkout_spec.rb`, писан като `expect { ... }.to raise_error(...)`, вече не
съответстваше на това как реално се държи правилно конфигурирано Rails приложение. Проверяваното
свойство за сигурност не се промени (`Checkout::OrdersController#show` все още се разрешава строго
през `Current.user.orders.find(...)`); само assertion-ът беше обновен, до `expect(response).to
have_http_status(:not_found)`.

Финално, реално изпълнение:

```text
$ docker compose run --rm web bundle exec rspec
.................FFFFF

Finished in 3 minutes 36 seconds (files took 13.38 seconds to load)
22 examples, 5 failures

Failed examples:

rspec ./spec/system/accessibility_spec.rb:21 # ... has no automatically detectable WCAG 2.1 AA violations on the catalog index
rspec ./spec/system/accessibility_spec.rb:28 # ... has no automatically detectable WCAG 2.1 AA violations on a product page
rspec ./spec/system/seo_spec.rb:21 # ... renders a unique, descriptive <title> per page
rspec ./spec/system/seo_spec.rb:32 # ... renders valid Product JSON-LD with the required merchant-listing fields
rspec ./spec/system/seo_spec.rb:46 # ... renders the four required Open Graph tags with an absolute og:image URL
```

**17/17 model и request специфициите (specs) минават — реално.** 5-те провала са всеки пример
`type: :system` (`spec/system/accessibility_spec.rb`, `spec/system/seo_spec.rb`), и всичките пет се
провалят идентично:

```text
Selenium::WebDriver::Error::WebDriverError:
  unable to connect to /root/.cache/selenium/chromedriver/.../chromedriver 127.0.0.1:9515:
  Errno::ECONNREFUSED: Failed to open TCP connection to 127.0.0.1:9515 (Connection refused ...)
```

Това е очаквано, не грешка: този Docker образ умишлено е без Node/браузър (§3 на заземяващата
бележка на този addendum — стекът Propshaft + importmap-rails на Rails 8.1 не се нуждае от Node, за
да боот-не или рендира ERB), а собствената, вече съществуваща секция „Frontend gate" на README-то
вече документира, че тези конкретни спекове се нуждаят от реален Chrome + `chromedriver` на `PATH`,
инсталирани отделно (`brew install --cask google-chrome`, `brew install chromedriver`) — точно
машинно-локалната настройка, която този минимален контейнер умишлено не носи. Инсталирането на
Chrome в образа, за да се направят тези 5 примера зелени, би противоречило на собствената инструкция
„запази го минимален" на addendum-а за boot образ; поправката, ако искаш тези 5 да се пускат и вътре
в Docker, е документирана в README.md §3 за нативния macOS път, не нещо, което този контейнер е
предназначен да възпроизвежда.

### Teardown — реален, чист

```text
$ docker compose down -v
 Container rails-estore-web-1  Removed
 Volume rails-estore_rails_storage  Removed
 Network rails-estore_default  Removed

$ docker rmi rails-estore-web
Untagged: rails-estore-web:latest
Deleted: sha256:...

$ docker compose ps -a
NAME   IMAGE   COMMAND   SERVICE   CREATED   STATUS   PORTS
(empty)
```

Нито един контейнер, volume или image не остана работещ или висящ след верификацията.

### Secret / key-shaped-string scan на Docker добавките — реален, чист

```text
$ grep -rInE '[a-z]k_(l[iv]{2}e|test)_[A-Za-z0-9]{10,}' code/rails-estore/
(no matches)
```

Няма `.env` файл в дървото (само commit-натия `.env.example`); `env_file: [{ path: .env, required:
false }]` на `docker-compose.yml` е толерантен към липса по дизайн — `docker compose up` никога не
се проваля само защото читателят не си е създал собствен `.env`.

## Обобщение

| Проверка | Резултат |
|---|---|
| `settings.json` е добре формиран JSON | PASS (реално) |
| 5× frontmatter на агентите добре формиран + валиден спрямо схемата | PASS (5/5, реално) |
| `context.sh` се изпълнява чисто | PASS (exit 0, реално — заснето наново на 2026-09-05 на български) |
| `guard.sh` пропуска безобидно, блокира `rm -rf /`, `--no-verify`, жив Stripe ключ, и отпечатване на тайна | PASS (5/5 случая, реално — заснето наново на 2026-09-05 на български) |
| `verify.sh` деградира елегантно без `bundle` | PASS (3/3 случая, реално — заснето наново на 2026-09-05 на български) |
| Secret scan през `code/rails-estore/` | PASS — чисто (реално, заснето наново на 2026-09-05) |
| `bundle exec rspec` / `rubocop` / `brakeman -q --no-summary` | **НЕ е изпълнено — няма Ruby/Rails toolchain в тази среда (sandbox).** Представено като заземена референция (точни команди, точни фиксирани версии); възпроизведи на машина с инсталирани Ruby 4.0.6 / Rails 8.1.3.1. |
| axe / html-proofer / `npx lhci autorun` (frontend гейт) | **НЕ е изпълнено — няма браузър/chromedriver/Node в тази среда (sandbox).** Представено като заземена референция (§6); възпроизведи по секцията „Frontend gate" на `README.md`. |
| `docker compose build` | PASS (реално, след поправка на реален конфликт между версии на rubocop/rubocop-rails) |
| `docker compose up` — `/up`, `/`, `/session/new`, `/products/1` | PASS — всички 200 (реално) |
| Страниците за каталог + вход визуално проверени (реален браузър срещу работещия контейнер) | PASS (реално — също улови и поправи реална грешка в рендиране на ERB коментар) |
| `docker compose run --rm web bundle exec rspec` | 17/17 model + request специфициите (specs) PASS (реално); 5/5 system специфициите се провалят поради липсващ chromedriver (очаквано — този образ умишлено е без браузър) |
| `docker compose down -v` + премахване на image | PASS — чист teardown, нищо не остана работещо (реално) |
| Secret/key-shaped-string scan на Docker добавките | PASS — чисто (реално) |

Дата на проверка: 2026-09-04 (§3 — изходът на hook-овете — заснет наново на 2026-09-05, след
превода на hook скриптовете на български; §5 презаснет наново на 2026-09-05 по същата причина).
