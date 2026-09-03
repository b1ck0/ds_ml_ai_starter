"""Unit tests for the Elders Tribunal orchestrator -- stdlib `unittest` only,
NO third-party test framework, NO network, NO API key.

Every elder and moderator below is a SCRIPTED FAKE: it returns fixed text from
a list, never calls an LLM. This is the same test double pattern as a Java
`FakeRepository implements Repository` used instead of a real JDBC-backed one
in a unit test -- the fake satisfies the `Elder` interface from elder.py, so
`Tribunal` (orchestrator.py) cannot tell it apart from a real, LLM-backed elder.

Run:
    .venv-agent/Scripts/python.exe -m unittest test_orchestrator.py -v
    .venv-agent/Scripts/python.exe test_orchestrator.py
"""
from __future__ import annotations

import unittest

from elder import Elder, Moderator, Statement
from orchestrator import Tribunal


class ScriptedElder(Elder):
    """Returns the next line from a fixed script each time `respond()` is
    called; repeats the last line if called more times than the script is long.
    `self.calls` lets a test assert an elder was (or was NOT) called again --
    the mechanism the early-stop test below relies on.
    """

    def __init__(self, name: str, script: list[str]) -> None:
        super().__init__(name)
        self._script = script
        self.calls = 0

    def respond(self, topic: str, history: list[Statement]) -> str:
        text = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return text


class ScriptedModerator(Moderator):
    """Deterministic stand-in for an LLM moderator: reports unanimity or the
    raw vote split, and records its own last call so a test can assert exactly
    what the orchestrator handed it.
    """

    def __init__(self, name: str = "Test-Moderator") -> None:
        super().__init__(name)
        self.last_call: tuple[str, list[Statement], dict[str, str | None]] | None = None

    def synthesize(
        self, topic: str, history: list[Statement], votes: dict[str, str | None]
    ) -> str:
        self.last_call = (topic, list(history), dict(votes))
        cast = [v for v in votes.values() if v is not None]
        if cast and len(set(cast)) == 1:
            return f"Consensus reached on '{topic}': {cast[0]}."
        return f"No consensus on '{topic}'. Votes: {votes}."


class TestTurnTakingAndTermination(unittest.TestCase):
    """LO1 / LO2: round-robin turn order, growing history, and both stop
    conditions (unanimous early stop vs. max_rounds reached).
    """

    def test_round_robin_order_and_history_growth(self) -> None:
        elder_a = ScriptedElder("Elder-A", ["Point A1. VERDICT: YES", "Point A2. VERDICT: YES"])
        elder_b = ScriptedElder("Elder-B", ["Point B1. VERDICT: NO", "Point B2. VERDICT: YES"])
        tribunal = Tribunal([elder_a, elder_b], ScriptedModerator(), max_rounds=2)

        result = tribunal.convene("Is TDD worth the overhead for this team?")

        self.assertEqual(result.rounds_run, 2)
        self.assertEqual(len(result.transcript), 4)
        # Fixed round-robin order: A then B, every round, no exceptions.
        self.assertEqual(
            [s.speaker for s in result.transcript],
            ["Elder-A", "Elder-B", "Elder-A", "Elder-B"],
        )
        # Round numbers on the transcript match which round each turn was in.
        self.assertEqual([s.round_no for s in result.transcript], [1, 1, 2, 2])

    def test_second_speaker_sees_first_speakers_statement_same_round(self) -> None:
        seen_history_lengths: list[int] = []

        class RecordingElder(Elder):
            def __init__(self, name: str) -> None:
                super().__init__(name)

            def respond(self, topic: str, history: list[Statement]) -> str:
                seen_history_lengths.append(len(history))
                return "no opinion yet"

        first = RecordingElder("First")
        second = RecordingElder("Second")
        Tribunal([first, second], ScriptedModerator(), max_rounds=1).convene("topic")

        # First elder opens with an empty history; second already sees turn 1.
        self.assertEqual(seen_history_lengths, [0, 1])

    def test_early_stop_on_unanimous_verdict_skips_remaining_rounds(self) -> None:
        elder_a = ScriptedElder("Elder-A", ["Agree. VERDICT: YES", "SHOULD NOT BE CALLED"])
        elder_b = ScriptedElder("Elder-B", ["Agree too. VERDICT: YES", "SHOULD NOT BE CALLED"])
        tribunal = Tribunal([elder_a, elder_b], ScriptedModerator(), max_rounds=5)

        result = tribunal.convene("Should we adopt trunk-based development?")

        self.assertEqual(result.stop_reason, "unanimous")
        self.assertEqual(result.rounds_run, 1)
        self.assertTrue(result.unanimous)
        # Round 2 was never reached: each elder was called exactly once.
        self.assertEqual(elder_a.calls, 1)
        self.assertEqual(elder_b.calls, 1)

    def test_max_rounds_reached_without_unanimity(self) -> None:
        elder_a = ScriptedElder("Elder-A", ["Reason A1. VERDICT: YES", "Reason A2. VERDICT: YES"])
        elder_b = ScriptedElder("Elder-B", ["Reason B1. VERDICT: NO", "Reason B2. VERDICT: NO"])
        tribunal = Tribunal([elder_a, elder_b], ScriptedModerator(), max_rounds=2)

        result = tribunal.convene("Should we rewrite the billing module?")

        self.assertEqual(result.stop_reason, "max_rounds")
        self.assertEqual(result.rounds_run, 2)
        self.assertFalse(result.unanimous)
        self.assertEqual(elder_a.calls, 2)
        self.assertEqual(elder_b.calls, 2)


class TestConsensusAggregation(unittest.TestCase):
    """LO4: voting (strict majority) and moderator synthesis, including
    disagreement / deadlock handling.
    """

    def test_majority_vote_wins_without_unanimity(self) -> None:
        elder_a = ScriptedElder("Elder-A", ["VERDICT: YES"])
        elder_b = ScriptedElder("Elder-B", ["VERDICT: YES"])
        elder_c = ScriptedElder("Elder-C", ["VERDICT: NO"])
        tribunal = Tribunal([elder_a, elder_b, elder_c], ScriptedModerator(), max_rounds=1)

        result = tribunal.convene("Should we adopt microservices?")

        self.assertEqual(result.winning_verdict, "YES")
        self.assertFalse(result.deadlock)
        self.assertFalse(result.unanimous)  # 2-of-3 is a majority, not unanimity

    def test_deadlock_when_no_verdict_holds_a_strict_majority(self) -> None:
        elder_a = ScriptedElder("Elder-A", ["VERDICT: YES"])
        elder_b = ScriptedElder("Elder-B", ["VERDICT: NO"])
        elder_c = ScriptedElder("Elder-C", ["VERDICT: UNCERTAIN"])
        moderator = ScriptedModerator()
        tribunal = Tribunal([elder_a, elder_b, elder_c], moderator, max_rounds=1)

        result = tribunal.convene("Should we rewrite the auth service in Rust?")

        self.assertIsNone(result.winning_verdict)
        self.assertTrue(result.deadlock)
        self.assertIn("No consensus", result.consensus)
        # The moderator was handed the real vote split, not told to hide it.
        assert moderator.last_call is not None
        _, _, votes_seen = moderator.last_call
        self.assertEqual(votes_seen, {"Elder-A": "YES", "Elder-B": "NO", "Elder-C": "UNCERTAIN"})

    def test_missing_verdict_is_treated_as_abstention_not_an_error(self) -> None:
        elder_a = ScriptedElder("Elder-A", ["I have no strong opinion on this one."])
        elder_b = ScriptedElder("Elder-B", ["VERDICT: YES"])
        tribunal = Tribunal([elder_a, elder_b], ScriptedModerator(), max_rounds=1)

        result = tribunal.convene("Should we pin transitive dependencies?")

        self.assertIsNone(result.votes["Elder-A"])
        self.assertEqual(result.votes["Elder-B"], "YES")
        # One abstention, one YES: YES has 1 of 2 -- not a strict majority.
        self.assertIsNone(result.winning_verdict)
        self.assertTrue(result.deadlock)

    def test_verdict_extraction_is_case_insensitive_and_ignores_surrounding_text(self) -> None:
        elder_a = ScriptedElder(
            "Elder-A",
            ["This is a long rebuttal with reasoning.\n\nverdict: yes\n"],
        )
        elder_b = ScriptedElder("Elder-B", ["Short. VERDICT: YES"])
        tribunal = Tribunal([elder_a, elder_b], ScriptedModerator(), max_rounds=1)

        result = tribunal.convene("topic")

        self.assertEqual(result.votes["Elder-A"], "YES")
        self.assertTrue(result.unanimous)


class TestTribunalConstruction(unittest.TestCase):
    """Constructor guards -- the protocol's basic invariants."""

    def test_requires_at_least_two_elders(self) -> None:
        with self.assertRaises(ValueError):
            Tribunal([ScriptedElder("Solo", ["VERDICT: YES"])], ScriptedModerator())

    def test_requires_at_least_one_round(self) -> None:
        elders = [ScriptedElder("A", ["x"]), ScriptedElder("B", ["y"])]
        with self.assertRaises(ValueError):
            Tribunal(elders, ScriptedModerator(), max_rounds=0)


if __name__ == "__main__":
    unittest.main()
