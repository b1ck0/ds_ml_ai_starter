"""The provider-agnostic Elder interface -- the one contract every debating
agent in this chapter implements, whether it is a scripted fake (see
test_orchestrator.py, run.py) or a real LLM behind an API key (providers.py).

Java analogy: this is exactly `interface Elder { String respond(String topic,
List<Statement> history); }` plus a sibling `interface Moderator`. The
orchestrator in orchestrator.py is written once, against these two interfaces,
and never against a concrete provider -- the same dependency-inversion shape as
coding a service against a `Repository` interface instead of a concrete JDBC
`DataSource`. Swapping which LLM sits behind an `Elder` is swapping the
implementation class, never touching the orchestrator.

This module has ZERO third-party imports on purpose: it is pure interface (plus
one small, immutable value type), so it never needs a provider SDK installed to
be imported, type-checked, or unit-tested.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Statement:
    """One turn in the debate: who said it, in which round, and the text itself.

    Immutable (frozen=True) for the same reason a Java value object is usually
    made immutable: a `Statement` is a historical fact once spoken -- nothing in
    the orchestrator should be able to mutate round 1's transcript while round 3
    is being decided.
    """

    speaker: str
    round_no: int
    text: str


class Elder(ABC):
    """One debating agent. Concrete elders differ only in how `respond()` is
    implemented -- everything about rounds, turn order, and consensus lives in
    the orchestrator, not here.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def respond(self, topic: str, history: list[Statement]) -> str:
        """Produce this elder's next statement.

        Args:
            topic: the question/proposition under debate. Fixed for the whole
                tribunal session.
            history: every statement made so far, in order, across ALL elders
                and ALL rounds (does not include the statement being formed
                right now). Empty on an elder's very first turn.

        Returns:
            This elder's statement text for the current turn. By convention
            (not enforced by this interface -- see orchestrator.py's docstring),
            a statement that wants to register a vote ends with a line of the
            exact form ``VERDICT: <TAG>`` (e.g. ``VERDICT: YES``). An elder that
            omits it is treated as abstaining for that round.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover -- cosmetic only
        return f"{type(self).__name__}(name={self.name!r})"


class Moderator(ABC):
    """The role that closes a tribunal session: reads the full transcript and
    the final vote tally, and writes the human-readable consensus report.

    Kept as a separate interface from `Elder` (rather than "just another
    elder") because its contract is genuinely different -- it does not argue a
    position across rounds, it synthesises one, once, at the end. A Java analogy:
    `Elder` and `Moderator` are two small interfaces a single class *could*
    implement both of, but usually shouldn't -- same instinct as keeping a
    `Reader` and a `Writer` interface separate even when one class happens to be
    both.
    """

    def __init__(self, name: str = "Moderator") -> None:
        self.name = name

    @abstractmethod
    def synthesize(
        self,
        topic: str,
        history: list[Statement],
        votes: dict[str, str | None],
    ) -> str:
        """Produce the final consensus report.

        Args:
            topic: the question/proposition under debate.
            history: the full transcript, every round, every elder.
            votes: each elder's name mapped to the verdict tag extracted from
                their FINAL round of statements, or ``None`` if that elder
                never emitted a ``VERDICT: ...`` line.

        Returns:
            A human-readable consensus report. Should explicitly say whether
            consensus was reached, and if not, describe the disagreement
            rather than silently picking a side (see orchestrator.py's
            deadlock handling).
        """
        raise NotImplementedError
