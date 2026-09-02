"""Runnable, MOCK-BACKED demo of the Elders Tribunal -- no network call, no API
key, no provider SDK import anywhere in this file. Every elder below is a
scripted fake (see PersonaElder), so this script's output is 100% deterministic
and reproducible -- run it twice, get byte-identical transcripts. Contrast this
with providers.py's Pitfalls note on real-LLM non-determinism.

Two debates run back to back:
  1. A three-elder panel that runs the full round cap and settles on a
     MAJORITY vote (not unanimous) -- demonstrates the voting side of
     consensus aggregation.
  2. A three-elder panel that DEADLOCKS (three-way split, no strict majority)
     -- demonstrates disagreement handling: the moderator reports the split
     honestly instead of forcing a winner.

Run:
    .venv-agent/Scripts/python.exe run.py
"""
from __future__ import annotations

from elder import Elder, Moderator, Statement
from orchestrator import Tribunal, format_report


class PersonaElder(Elder):
    """A scripted elder with a fixed, per-round script -- a stand-in for a real
    LLM-backed elder (see providers.py's AnthropicElder / OpenAIElder /
    GoogleElder, which implement the exact same `Elder` interface against a
    real API instead of a Python list).
    """

    def __init__(self, name: str, script: list[str]) -> None:
        super().__init__(name)
        self._script = script

    def respond(self, topic: str, history: list[Statement]) -> str:
        # How many times THIS elder has already spoken tells us which script
        # line comes next (capped at the last line if asked for more rounds
        # than the script has -- an elder "repeats its position").
        turns_taken = sum(1 for s in history if s.speaker == self.name)
        return self._script[min(turns_taken, len(self._script) - 1)]


class SynthesizingModerator(Moderator):
    """A scripted moderator: summarises the real vote tally with plain string
    formatting -- no LLM call. A real deployment would swap this for a
    provider-backed moderator (same `synthesize()` contract) that asks an LLM
    to write the prose; the vote-counting logic here would not change, because
    it is not the moderator's job -- see orchestrator.py's `_tally()`.
    """

    def synthesize(
        self, topic: str, history: list[Statement], votes: dict[str, str | None]
    ) -> str:
        cast = {name: v for name, v in votes.items() if v is not None}
        if cast and len(set(cast.values())) == 1:
            verdict = next(iter(cast.values()))
            return (
                f"Consensus: the tribunal unanimously reports '{verdict}' on "
                f"'{topic}' ({len(cast)}/{len(votes)} elders cast a verdict)."
            )
        tally: dict[str, int] = {}
        for verdict in cast.values():
            tally[verdict] = tally.get(verdict, 0) + 1
        ordered = sorted(tally.items(), key=lambda item: -item[1])
        breakdown = ", ".join(f"{verdict}={count}" for verdict, count in ordered)
        if ordered and ordered[0][1] > len(votes) / 2:
            return (
                f"Consensus by majority: '{ordered[0][0]}' on '{topic}' "
                f"({breakdown}, out of {len(votes)} elders)."
            )
        return (
            f"No consensus on '{topic}' -- the panel is split ({breakdown}). "
            "Recommending the team gather more evidence or escalate to a human "
            "decision-maker rather than forcing a false majority."
        )


def run_majority_debate() -> None:
    pragmatist = PersonaElder(
        "Pragmatist-Elder",
        [
            "Ship the monolith first -- microservices add operational overhead "
            "this 4-person team hasn't earned yet. VERDICT: NO",
            "Even after hearing the scalability case, the coordination cost of "
            "running N services outweighs the benefit for a team this size "
            "today. VERDICT: NO",
            "Position unchanged: revisit this once the team has grown or "
            "traffic data justifies the split. VERDICT: NO",
        ],
    )
    architect = PersonaElder(
        "Architect-Elder",
        [
            "Start with clear module boundaries inside one deployable; split "
            "into services only once a boundary proves it needs to scale "
            "independently. VERDICT: NO",
            "Agreed with Pragmatist-Elder -- premature service boundaries are "
            "harder to undo than a well-modularised monolith. VERDICT: NO",
            "Holding this position through the final round. VERDICT: NO",
        ],
    )
    scale_advocate = PersonaElder(
        "Scale-Elder",
        [
            "If the projected growth is real, retrofitting service boundaries "
            "later costs more than designing them in from day one. VERDICT: YES",
            "I hear the team-size argument, but I still think the retrofit "
            "cost is being understated. Sticking with my position for now. "
            "VERDICT: YES",
            "Still YES, though I'll concede this is now a minority view on "
            "this panel. VERDICT: YES",
        ],
    )
    tribunal = Tribunal(
        elders=[pragmatist, architect, scale_advocate],
        moderator=SynthesizingModerator("Moderator"),
        max_rounds=3,
    )
    topic = "Should this 4-engineer greenfield team start with microservices?"
    result = tribunal.convene(topic)
    print(format_report(result))


def run_deadlock_debate() -> None:
    rewrite_now = PersonaElder(
        "Elder-A", ["Rewrite the auth service in Rust this quarter. VERDICT: YES"]
    )
    keep_python = PersonaElder(
        "Elder-B", ["Keep it in Python -- the risk isn't worth it right now. VERDICT: NO"]
    )
    prototype_first = PersonaElder(
        "Elder-C",
        ["Neither yet -- prototype both for two weeks before deciding. VERDICT: UNCERTAIN"],
    )
    tribunal = Tribunal(
        elders=[rewrite_now, keep_python, prototype_first],
        moderator=SynthesizingModerator("Moderator"),
        max_rounds=1,
    )
    topic = "Should we rewrite the auth service in Rust this quarter?"
    result = tribunal.convene(topic)
    print(format_report(result))


def main() -> None:
    print("=" * 72)
    print("DEBATE 1 -- three elders, full round cap, majority (not unanimous)")
    print("=" * 72)
    run_majority_debate()
    print()
    print("=" * 72)
    print("DEBATE 2 -- three-way split: deadlock, no forced consensus")
    print("=" * 72)
    run_deadlock_debate()


if __name__ == "__main__":
    main()
