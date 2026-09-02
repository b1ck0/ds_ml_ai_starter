"""The typed invoice schema -- the contract every extractor must satisfy.

Whichever extractor produces the raw fields (extract.py's rule-based fallback, or
the key-gated LLM path), neither one is trusted until it has been coerced through
this pydantic model. That is the whole point of the exercise: an LLM (or a regex)
only ever produces text/JSON-shaped data: nothing stops it from hallucinating a
field, dropping one, or returning "twelve dollars" where a number was expected.
Pydantic is the boundary that turns "probably-shaped data" into "guaranteed-shaped
data" -- the same role a `record Invoice(...)` with Bean Validation `@NotNull` /
`@Positive` annotations plays in Java, except the checks run at *construction* time
(`Invoice(**raw)` either returns a fully valid instance or raises), not later when
some downstream code finally dereferences a null.

pydantic 2.13.5 (installed in .venv-agent; confirmed as the current release on PyPI,
released 2026-08-28) [source: PyPI](https://pypi.org/project/pydantic/) (checked
2026-09-03). `field_validator` / `model_validator` are the current v2 decorator APIs
[source: pydantic docs](https://docs.pydantic.dev/latest/concepts/validators/)
(checked 2026-09-03).

Run this file directly for a couple of quick, no-key sanity checks:
    .venv-agent/Scripts/python.exe schema.py
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A cent of slack for float/rounding drift between an extractor's arithmetic and the
# printed subtotal/total on the invoice (e.g. a vendor's PDF generator rounding each
# line independently). Tighten this if your domain requires exact reconciliation.
_CENT = Decimal("0.01")
_TOLERANCE = Decimal("0.02")


def _round_cents(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


class LineItem(BaseModel):
    """One row of the invoice's line-item table."""

    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    amount: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def check_amount_matches_quantity_times_price(self) -> "LineItem":
        expected = _round_cents(Decimal(self.quantity) * self.unit_price)
        if abs(expected - self.amount) > _TOLERANCE:
            raise ValueError(
                f"line item {self.description!r}: quantity ({self.quantity}) * "
                f"unit_price ({self.unit_price}) = {expected}, but amount on the "
                f"invoice is {self.amount} -- extraction likely misread a column"
            )
        return self


class Invoice(BaseModel):
    """A validated invoice: the shape every extractor (rule-based or LLM) must fill.

    Field-level validators catch malformed individual values (a badly-formed invoice
    number, a negative total); the model-level validator below catches an
    internally-inconsistent document (a subtotal that doesn't match its own line
    items) -- the kind of error a single field's type hint can never express.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    vendor_name: str = Field(min_length=1)
    vendor_address: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    bill_to: str = Field(min_length=1)
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal = Field(ge=0)
    tax: Decimal = Field(ge=0)
    total: Decimal = Field(ge=0)

    @field_validator("invoice_number")
    @classmethod
    def invoice_number_looks_like_one(cls, value: str) -> str:
        # Deliberately loose (no vendor numbering scheme is universal) but rejects
        # the clearest sign of a bad extraction: an empty or whitespace-only string.
        if not value.strip():
            raise ValueError("invoice_number must not be blank")
        return value

    @field_validator("invoice_date", mode="before")
    @classmethod
    def parse_iso_date(cls, value: object) -> object:
        # Accept a plain "YYYY-MM-DD" string (what both extractors in this chapter
        # produce) as well as an already-parsed date -- pydantic would otherwise
        # only accept ISO 8601 strings or date/datetime objects here by default.
        if isinstance(value, str):
            return date.fromisoformat(value.strip())
        return value

    @model_validator(mode="after")
    def check_totals_reconcile(self) -> "Invoice":
        computed_subtotal = _round_cents(sum((item.amount for item in self.line_items), Decimal("0")))
        if abs(computed_subtotal - self.subtotal) > _TOLERANCE:
            raise ValueError(
                f"subtotal on invoice ({self.subtotal}) does not match the sum of "
                f"line item amounts ({computed_subtotal})"
            )
        computed_total = _round_cents(self.subtotal + self.tax)
        if abs(computed_total - self.total) > _TOLERANCE:
            raise ValueError(
                f"total on invoice ({self.total}) does not equal subtotal + tax "
                f"({self.subtotal} + {self.tax} = {computed_total})"
            )
        return self


if __name__ == "__main__":
    # A minimal, valid instance -- proves the model_validator's arithmetic checks
    # pass on a document that actually reconciles.
    ok = Invoice(
        vendor_name="ACME WIDGETS INC.",
        vendor_address="123 Foundry Lane, Springfield, IL 62701",
        invoice_number="INV-2026-0042",
        invoice_date="2026-08-15",
        bill_to="Contoso Ltd, 77 Market St, Boston MA 02108",
        line_items=[
            LineItem(description="Widget A", quantity=10, unit_price="12.50", amount="125.00"),
        ],
        subtotal="125.00",
        tax="10.00",
        total="135.00",
    )
    print("valid invoice constructed:", ok.invoice_number, ok.total)

    # A deliberately broken one -- total doesn't equal subtotal + tax.
    try:
        Invoice(
            vendor_name="ACME WIDGETS INC.",
            vendor_address="123 Foundry Lane, Springfield, IL 62701",
            invoice_number="INV-2026-0043",
            invoice_date="2026-08-15",
            bill_to="Contoso Ltd, 77 Market St, Boston MA 02108",
            line_items=[
                LineItem(description="Widget A", quantity=10, unit_price="12.50", amount="125.00"),
            ],
            subtotal="125.00",
            tax="10.00",
            total="999.00",
        )
    except Exception as exc:  # pydantic.ValidationError
        print("rejected as expected:", type(exc).__name__)
        print(exc)
