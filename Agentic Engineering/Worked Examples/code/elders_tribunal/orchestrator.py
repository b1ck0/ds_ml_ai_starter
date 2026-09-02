"""The Elders Tribunal orchestration engine: the multi-agent PROTOCOL, decoupled
from any concrete LLM.

## The protocol (LO1)

**Roles.** A fixed panel of >=2 `Elder`s (ideally each backed by a different LLM
provider -- see providers.py) plus exactly one `Moderator`. Elders argue; the
moderator never argues, it only synthesises the panel's final report.

**Turn order.** Round-robin, in the exact order the elders were passed to
`Tribunal(...)`. Round 1, elder[0] speaks first with an EMPTY history (it has
seen nothing yet -- it is opening the debate); elder[1] speaks next and already
sees elder[0]'s round-1 statement; and so on. Round 2 repeats the same order,
but now every elder sees the FULL history from round 1. This is deliberately the
simplest fair turn order there is -- a fixed queue, not a scheduler -- so the
protocol stays legible enough to trace by hand (SPEC-AGENT-5's "keep it
minimal" scope note).

**The vote convention.** Every elder statement MAY end with a line of the exact
form ``VERDICT: <TAG>`` (see elder.py's `Elder.respond` docstring). This module
never inspects statement text for anything else -- no sentiment analysis, no
keyword matching -- so the entire termination and voting mechanism is driven by
one regex and nothing more.

**Termination (two independent conditions, whichever fires first):**
1. *Unanimous early stop* -- after any round in which every elder cast a verdict
   and all of them match, the tribunal stops immediately, even if rounds remain.
   No point paying for (or waiting on) further rounds once everyone agrees.
2. *Max rounds* -- a hard cap (`max_rounds`, constructor argument) the tribunal
   never exceeds, so a panel that keeps disagreeing cannot debate forever. This
   is the same role a timeout or a max-retry count plays in a Java retry loop:
   an upper bound that guarantees the process terminates.

**Consensus aggregation (LO4).** Two techniques, both always run:
- *Voting*: `_tally()` counts the FINAL round's verdicts and reports a winner
  only if it holds a strict majority (`> len(elders) / 2`); otherwise there is
  no numerical winner (`winning_verdict is None`) and `deadlock=True`.
- *Moderator synthesis*: `moderator.synthesize(...)` always runs too, and is
  told the vote tally -- so it can announce "unanimous", "majority", or
  explicitly "no consensus, here is the split" (Section 5 of the chapter shows
  a scripted moderator doing exactly this). Voting gives you a machine-checkable
  number; synthesis gives you the human-readable reason behind it. Neither
  replaces the other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from elder import Elder, Moderator, Statement

# Matches a line like "VERDICT: YES" (case-insensitive, tag is one word).
_VERDICT_RE = re.compile(r"VERDICT:\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def extract_verdict(statement_text: str) -> str | None:
    """Pull the ``VERDICT: <TAG>`` line's tag out of a statement, upper-cased.

    Returns None if the statement never registered a verdict -- treated as an
    abstention everywhere else in this module, never as an error: an elder is
    allowed to speak without voting on every single turn.
    """
    match = _VERDICT_RE.search(statement_text)
    return match.group(1).upper() if match else None


def _tally(votes: dict[str, str | None], total_elders: int) -> tuple[str | None, bool]:
    """Count the final-round verdicts. Returns (winning_verdict_or_None, deadlock).

    A verdict wins only with a STRICT majority (more than half of all elders,
    not just a plurality of those who voted) -- so a 3-way split among 3 elders,
    or a 2-2 tie among 4, is correctly reported as a deadlock rather than
    letting whichever tag happened to be counted first "win".
    """
    counts: dict[str, int] = {}
    for verdict in votes.values():
        if verdict is None:
            continue
        counts[verdict] = counts.get(verdict, 0) + 1
    if not counts:
        return None, True
    winner, top_count = max(counts.items(), key=lambda item: item[1])
    has_majority = top_count > total_elders / 2
    return (winner if has_majority else None), (not has_majority)


@dataclass
class DebateResult:
    """Everything the tribunal produced -- the full transcript plus the outcome."""

    topic: str
    transcript: list[Statement] = field(default_factory=list)
    rounds_run: int = 0
    stop_reason: str = "max_rounds"  # "unanimous" | "max_rounds"
    votes: dict[str, str | None] = field(default_factory=dict)
    winning_verdict: str | None = None
    unanimous: bool = False
    deadlock: bool = False
    consensus: str = ""


class Tribunal:
    """Runs the protocol described in this module's docstring against a fixed
    panel of `Elder`s and one `Moderator`.

    The orchestrator depends only on `elder.Elder` / `elder.Moderator` -- it
    never imports, or knows about, `providers.py`. That is what makes it
    unit-testable with plain, scripted fakes (test_orchestrator.py) and lets the
    exact same class later run a real, key-gated, multi-provider debate without
    a single line of this file changing.
    """

    def __init__(self, elders: list[Elder], moderator: Moderator, max_rounds: int = 3) -> None:
        if len(elders) < 2:
            raise ValueError("a tribunal needs at least two elders to debate")
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._elders = list(elders)
        self._moderator = moderator
        self._max_rounds = max_rounds

    def convene(self, topic: str) -> DebateResult:
        """Run the full protocol for one topic and return the result.

        This method is deterministic given deterministic elders (as every
        scripted fake in this chapter is) -- the only source of non-determinism
        in a real deployment is inside a real LLM's `respond()` implementation
        (see providers.py and the chapter's Pitfalls section on temperature).
        """
        history: list[Statement] = []
        stop_reason = "max_rounds"
        rounds_run = 0
        round_verdicts: dict[str, str | None] = {}

        for round_no in range(1, self._max_rounds + 1):
            rounds_run = round_no
            round_verdicts = {}
            for elder in self._elders:
                # Each elder sees the FULL history so far, including every
                # statement made earlier in this same round by elders ahead of
                # it in turn order -- that is what makes later speakers able to
                # rebut, and is exactly what "round-robin, full history" means.
                text = elder.respond(topic, list(history))
                history.append(Statement(speaker=elder.name, round_no=round_no, text=text))
                round_verdicts[elder.name] = extract_verdict(text)

            cast = [v for v in round_verdicts.values() if v is not None]
            everyone_voted = len(cast) == len(self._elders)
            if everyone_voted and len(set(cast)) == 1:
                stop_reason = "unanimous"
                break

        votes = round_verdicts
        winning_verdict, deadlock = _tally(votes, total_elders=len(self._elders))
        unanimous = stop_reason == "unanimous"
        consensus = self._moderator.synthesize(topic, history, votes)

        return DebateResult(
            topic=topic,
            transcript=history,
            rounds_run=rounds_run,
            stop_reason=stop_reason,
            votes=votes,
            winning_verdict=winning_verdict,
            unanimous=unanimous,
            deadlock=deadlock,
            consensus=consensus,
        )


def format_report(result: DebateResult) -> str:
    """Render a `DebateResult` as the human-readable transcript + verdict block
    used by run.py. Pure formatting -- no logic that belongs in `Tribunal`.
    """
    lines = [
        f"Topic: {result.topic}",
        f"Rounds run: {result.rounds_run} (stop reason: {result.stop_reason})",
        "",
        "--- Transcript ---",
    ]
    for statement in result.transcript:
        lines.append(f"[Round {statement.round_no}] {statement.speaker}: {statement.text}")
    lines.append("")
    lines.append("--- Votes (final round) ---")
    for name, verdict in result.votes.items():
        lines.append(f"  {name}: {verdict if verdict is not None else 'ABSTAIN'}")
    lines.append("")
    lines.append(
        f"Unanimous: {result.unanimous}   Deadlock: {result.deadlock}   "
        f"Winning verdict: {result.winning_verdict}"
    )
    lines.append("")
    lines.append("--- Consensus (moderator) ---")
    lines.append(result.consensus)
    return "\n".join(lines)
