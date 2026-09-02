# FEATURE-1: Reject invalid card numbers before they reach the payment gateway

**Status:** approved
**Owner:** architect (Opus)
**Routing:** implementer=Sonnet · research=none needed (see below) · review=Sonnet (fresh)

## Intent

Support has traced three chargeback disputes this month to card numbers that were typo'd at
checkout — a digit transposed, one dropped — and never should have reached the payment gateway at
all. Add a validator that rejects a candidate card number before we spend a gateway round-trip (and
risk a decline fee) finding out it was never valid. The industry-standard, gateway-independent first
check for this is the Luhn checksum
[source: Luhn algorithm — Wikipedia](https://en.wikipedia.org/wiki/Luhn_algorithm) (checked
2026-09-02) — every major card scheme's PAN (primary account number) is constructed to satisfy it,
so a PAN that fails it is provably malformed, no gateway call required.

## Acceptance criteria

- AC1 — `LuhnValidator.isValid(String)` returns `true` for a digit string whose Luhn checksum is
  valid, and accepts spaces and hyphens as separators (e.g. `"4012 8888 8888 1881"`).
- AC2 — returns `false` for a digit string whose checksum is invalid.
- AC3 — returns `false` for a non-empty string containing a non-digit character (after stripping
  spaces/hyphens), and for an empty string — it does not throw for these; a malformed candidate is
  simply "not valid," not an error condition.
- AC4 — throws `IllegalArgumentException` for a `null` candidate — a `null` reaching this method is
  a caller bug (a missing field, not a malformed card number), and that is a different failure mode
  than "the user mistyped a digit."

## Claims to ground

None required. The Luhn checksum is a fixed, publicly defined arithmetic algorithm (not a package
version, a dataset, or a "library X does Y" claim) — grounded here directly by the algorithm
definition and an inline citation with the date checked, per this project's grounding rule
(`CLAUDE.md` golden rule 3). No dependency is added by this feature beyond the JUnit 5 test
dependency already pinned in `pom.xml`.

## Out of scope

- Card-network identification (Visa/Mastercard/Amex prefix + length rules) — a separate feature if
  ever needed.
- Calling the payment gateway at all — this validator is a pre-flight check only.

## Assets to produce

- `src/main/java/com/example/sdlcdemo/LuhnValidator.java`
- `src/test/java/com/example/sdlcdemo/LuhnValidatorTest.java`

## Gates

Entry: this spec approved. Exit: `docs/definition-of-done.md` checklist, in full.
