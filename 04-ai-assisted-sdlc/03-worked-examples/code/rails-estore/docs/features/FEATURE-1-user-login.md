# FEATURE-1: Регистрация, вход и изход на потребител

**Status:** approved
**Owner:** архитект (Opus)
**Routing:** implementer=Sonnet · research=NOTE-SDLC-4-1, NOTE-SDLC-4-2, NOTE-SDLC-4-5 · review=Sonnet (fresh)

## Намерение (Intent)

Преди да може да има количка или checkout, трябва да има акаунт. Дай на посетител начин да се
регистрира, да влезе и да излезе, използвайки формата на нативния authentication generator на
Rails 8 (`User` + `has_secure_password`, `Session`, проследена в базата данни, `Current`), вместо
външен gem — леката на зависимости, официално поддържана опция за ново Rails 8 приложение
[source: NOTE-SDLC-4-2-auth-generator.md]. Самият generator не доставя flow за регистрация
[source: NOTE-SDLC-4-2-auth-generator.md, "Caveats"] — тази функционалност добавя
`RegistrationsController` върху него.

## Критерии за приемане (Acceptance criteria)

- AC1 — `POST /registration` с валидни `email_address`, `password`, и `password_confirmation`
  създава `User`, стартира сесия (ред `Session` + подписана постоянна бисквитка), и пренасочва към
  root path.
- AC2 — колоната `password_digest` на създадения `User` никога не е равна на, и не може да бъде
  обърната обратно към, изпратената парола в чист текст — `has_secure_password` я хешира с bcrypt
  [source: NOTE-SDLC-4-5-auth-security.md].
- AC3 — `POST /registration` с допълнителен, непозволен (non-permitted) параметър (`admin: "true"`)
  създава потребителя БЕЗ да задава `admin` — списъкът `permit()` на registration controller-а не го
  включва [source: NOTE-SDLC-4-5-auth-security.md, mass-assignment].
- AC4 — `POST /session` с правилни credentials за съществуващ потребител го вписва (създаден ред
  `Session`, зададена бисквитка, пренасочване към страницата, която е опитвал да достигне, или root);
  с грешни credentials рендира отново формата за вход с грешка и НЕ създава сесия.
- AC5 — `DELETE /session` унищожава текущия ред `Session` и изчиства бисквитката.
- AC6 — заявка от изписан (signed-out) потребител към страница, изискваща автентикация
  (`GET /cart`), пренасочва към `new_session_path`, вместо да рендира страницата — чрез before-action
  `require_authentication` на concern-а `Authentication`, приложен по подразбиране в
  `ApplicationController` [source: NOTE-SDLC-4-2-auth-generator.md].

## Твърдения за заземяване (Claims to ground)

- Формата на нативния auth generator на Rails 8 (User/Session/Current, has_secure_password,
  подписана бисквитка) — заземено, `docs/research/NOTE-SDLC-4-2-auth-generator.md`.
- Версия и CVE статус на bcrypt, поведението на хеширане на `has_secure_password` — заземено,
  `docs/research/NOTE-SDLC-4-1-versions.md` и `docs/research/NOTE-SDLC-4-5-auth-security.md`.
- Strong parameters като защита срещу mass assignment — заземено,
  `docs/research/NOTE-SDLC-4-5-auth-security.md`.

## Извън обхват (Out of scope)

- Нулиране на парола (`PasswordsMailer` на generator-а съществува, но окабеляването на реален
  mailer/SMTP е отделна функционалност).
- Потвърждение на имейл / confirmable — не е част от нативния generator; територия на Devise, ако
  някога е нужно (дневник на решенията в `docs/architecture.md`).
- Rate-limiting на опити за вход отвъд документирания action helper `rate_limit` на Rails —
  hardening подход, не тази функционалност.

## Активи за изготвяне (Assets to produce)

- `app/models/user.rb`, `app/models/session.rb`, `app/models/current.rb`
- `app/controllers/concerns/authentication.rb`
- `app/controllers/application_controller.rb`
- `app/controllers/sessions_controller.rb`, `app/controllers/registrations_controller.rb`
- `app/views/sessions/new.html.erb`, `app/views/registrations/new.html.erb`
- `db/schema.rb` (таблиците users, sessions)
- `spec/models/user_spec.rb`, `spec/requests/registrations_spec.rb`, `spec/requests/sessions_spec.rb`

## Гейтове (Gates)

Вход: тази спецификация одобрена; трите бележки за заземяване по-горе пристигнали. Изход:
чеклистът `docs/definition-of-done.md`, изцяло — включително случаите за сигурност mass assignment и
хеширане на пароли (AC2, AC3), с по един изричен RSpec пример за всеки.
