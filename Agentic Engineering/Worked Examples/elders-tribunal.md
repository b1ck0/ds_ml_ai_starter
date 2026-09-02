# Elders Tribunal — A Multi-Agent Debate That Reports Consensus

*Agentic Engineering · Worked Examples · [SPEC-AGENT-5](../../specs/SPEC-AGENT-5-elders-tribunal.md)*

Every agent you have built in this course so far is one model, alone, deciding what to do. This
chapter builds something structurally different: several agents — **elders** — that each hold
their own position on a question, argue it across multiple rounds while reading what the others
just said, and hand the transcript to a **moderator** that reports whether the panel actually
agrees. Nothing here queries a database or reads a PDF; the entire subject is the *protocol*
multiple agents use to reach — or fail to reach — a shared answer.

If you have ever sat in an architecture review with three senior engineers who disagree, you
already know why this is worth building deliberately instead of hoping one model gets it right on
the first try: a single LLM call gives you one opinion, stated with the same confident tone whether
it is right or badly wrong. A panel that has to *state a position, defend it against pushback, and
either converge or admit it didn't* is a cheap, mechanical way to surface exactly the disagreement a
single call would hide.

## 1. What & why — the panel-of-experts framing

Picture the Java equivalent of "just call the model and print the answer": a single service method
that returns `String recommendation()`. It compiles, it runs, and it gives you no way to tell a
well-reasoned answer from a plausible-sounding guess — because there is only one code path and it
never has to defend itself.

Now picture a **panel**: several implementations of the same interface, each queried independently,
their outputs compared, and a designated reviewer writing up where they agreed and where they
didn't. That is not an ML idea — it is the same instinct behind running a critical decision past
more than one senior engineer instead of one, or behind an ensemble of classifiers in classical ML
(you likely met bagging/boosting's "many weak opinions, one aggregated answer" idea in
[SPEC-DS-8](../../specs/SPEC-DS-8-class-imbalance-undersampling-ensembles.md) if you took the Data
Science track). Applied to LLM agents, the same idea buys you two concrete things a single call
cannot:

1. **Different failure modes cancel out.** Different providers — trained on different data, tuned
   with different methods — are wrong in different ways. A question where Claude and GPT-5 land on
   opposite answers is a genuine signal that the question is harder than it looked; a question where
   they agree independently is much better evidence than either agreeing with itself twice.
2. **A visible chain of reasoning, not just a verdict.** The transcript in
   Section 5 is the artefact — it shows *how* the panel got from three opening positions to one
   report, not just the final line.

This chapter builds that pattern with a hard constraint from the spec: the multi-agent *logic* — who
speaks when, how many rounds run, how a stalemate is detected and reported — has to be fully
unit-testable **without a single API key**, and the real, different-LLM wiring has to sit in its own
clearly separate module. Section 3 builds the testable core; Section 4 wires the real providers
behind the exact same interface, key-gated.

**Environment for this chapter** (same dedicated environment as the rest of Agentic Engineering —
see [the MCP chapter's environment box](mcp-database-query-layer.md#2-the-database--a-seeded-deterministic-sqlite-file)):

```text
Python 3.13.7      (.venv-agent, confirmed while writing this chapter, 2026-09-03)
```

The runnable core (`elder.py`, `orchestrator.py`, `test_orchestrator.py`, `run.py`) uses **only the
Python standard library** — `abc`, `dataclasses`, `re`, `unittest`. No `pip install` is required to
run any of it, and nothing in `.venv-agent` was modified to write this chapter. `providers.py`
additionally needs three third-party SDKs, but only if you actually construct one of its classes —
Section 4 covers exactly what, and why it's safe to leave uninstalled.

## 2. The protocol — roles, rounds, turn order, stop condition (LO1)

Before any code, the rules the whole chapter implements:

**Roles.** A fixed panel of two or more **elders** (`Elder`, one per debating agent — ideally each
backed by a different LLM provider, Section 4) plus exactly one **moderator** (`Moderator`). Elders
argue a position across rounds; the moderator never argues — it only reads the finished transcript
and the final vote tally, once, and writes the consensus report.

**Turn order.** Strict round-robin, in the exact order the elders were constructed with. Round 1,
elder 1 opens with an *empty* history (nothing has been said yet); elder 2 speaks next and already
sees elder 1's opening statement; and so on down the panel. Round 2 repeats the same order, but now
every elder sees the *entire* round-1 transcript, from every elder, before it speaks again. This is
deliberately the simplest fair turn order that exists — a fixed queue, not a scheduler that decides
who speaks next based on content — which keeps the protocol something you can trace by hand instead
of a black box (SPEC-AGENT-5 scopes out "elaborate agent frameworks" on purpose).

**The vote convention.** By convention only — nothing in the `Elder` interface enforces it — a
statement that wants to register a position ends with a line of the exact form `VERDICT: <TAG>`
(e.g. `VERDICT: YES`). An elder that omits it for a given turn is treated as **abstaining**, not as
an error. This one convention is the entire mechanism both the stop condition and the vote tally are
built on.

**Stop condition — two independent checks, whichever fires first:**

- **Unanimous early stop.** After any round in which every elder cast a verdict and all of them
  match, the debate ends immediately, even with rounds left on the clock — there is no reason to pay
  for (or wait on) further rounds once the whole panel agrees.
- **Max rounds.** A hard cap the tribunal never exceeds. This plays the same role a timeout or a
  bounded retry count plays in a Java resilience pattern: an upper bound that *guarantees*
  termination even when the panel keeps disagreeing.

Here is that full protocol as an ASCII flow, traced against the real Debate 1 run from Section 5 —
matplotlib is not installed in `.venv-agent` (confirmed while writing this chapter), so this is
rendered as text, the same choice the MCP chapter made for its own sequence diagram
([mcp_query_sequence_diagram.txt](artefacts/mcp_query_sequence_diagram.txt)):

```text
                              Tribunal.convene(topic)
                                       |
                                       v
                    +----------------------------------------+
                    |  round_no = 1 .. max_rounds              |
                    |  (round-robin, fixed elder order)         |
                    +----------------------------------------+
                                       |
              for each elder, IN ORDER, within this round:
                                       |
                                       v
      +--------------------------------------------------------------+
      |  elder.respond(topic, history_so_far)                        |
      |    - history_so_far includes every statement made in EARLIER |
      |      rounds AND by earlier elders already in THIS round       |
      |    - elder appends a Statement(speaker, round_no, text)        |
      |      to the shared transcript                                 |
      |    - text MAY end with "VERDICT: <TAG>" (else = abstain)       |
      +--------------------------------------------------------------+
                                       |
                                       v
                    after the round: extract_verdict() on every
                    statement made THIS round, then:
                      - all voted AND all match  -> stop "unanimous"
                      - round_no == max_rounds    -> stop "max_rounds"
                      - otherwise                 -> next round
                                       |
                                       v
                    _tally(votes) -> (winning_verdict, deadlock)
                    moderator.synthesize(topic, history, votes)
                                       |
                                       v
                          DebateResult(transcript, votes,
                            winning_verdict, unanimous,
                            deadlock, consensus)
```

Full diagram, with both termination paths annotated against the real captured run:
[artefacts/debate_flow_diagram.txt](artefacts/debate_flow_diagram.txt).

## 3. The engine — a provider-agnostic `Elder` interface, unit-tested with fakes (LO2)

### 3.1 The interface

`elder.py` defines the entire contract the orchestrator depends on, and nothing else — it has zero
third-party imports, so it never needs a provider SDK installed just to exist:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Statement:
    """One turn in the debate: who said it, in which round, and the text itself."""

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
    def respond(self, topic: str, history: list) -> str:
        """Produce this elder's next statement given the topic and the full
        debate so far. By convention, a statement that wants to register a
        vote ends with a line of the exact form 'VERDICT: <TAG>'.
        """
        raise NotImplementedError
```

If you have written a Java `interface Elder { String respond(String topic, List<Statement>
history); }` and one small `record Statement(String speaker, int roundNo, String text)`, you have
already written this — `ABC` + `@abstractmethod` is Python's version of an interface with one
required method, and `@dataclass(frozen=True)` gives `Statement` the same "immutable value object"
guarantee a Java `record` gives you: nothing downstream can mutate round 1's transcript while round 3
is being decided. The orchestrator (`orchestrator.py`) is written once, against `Elder` and a sibling
`Moderator` interface, and never imports a concrete provider — the same dependency-inversion shape as
coding a service against a `Repository` interface instead of a concrete `DataSource`. Swapping which
LLM sits behind an elder means swapping the implementation class; the orchestrator never changes.
Full file: [code/elders_tribunal/elder.py](code/elders_tribunal/elder.py).

### 3.2 The orchestrator

`orchestrator.py` implements exactly the protocol from Section 2. The verdict-extraction regex and
the vote tally are the two pieces of logic everything else depends on:

```python
import re

_VERDICT_RE = re.compile(r"VERDICT:\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def extract_verdict(statement_text: str) -> str | None:
    """Pull the 'VERDICT: <TAG>' line's tag out of a statement, upper-cased.
    Returns None (treated as an abstention) if the statement never registered one.
    """
    match = _VERDICT_RE.search(statement_text)
    return match.group(1).upper() if match else None


def _tally(votes: dict, total_elders: int):
    """Count the final round's verdicts. A verdict wins only with a STRICT
    majority (more than half of ALL elders, not just of those who voted) --
    so a 3-way split, or a 2-2 tie among 4 elders, correctly reports as a
    deadlock instead of letting whichever tag was counted first "win".
    """
    counts: dict = {}
    for verdict in votes.values():
        if verdict is None:
            continue
        counts[verdict] = counts.get(verdict, 0) + 1
    if not counts:
        return None, True
    winner, top_count = max(counts.items(), key=lambda item: item[1])
    has_majority = top_count > total_elders / 2
    return (winner if has_majority else None), (not has_majority)
```

And the loop that drives the whole protocol:

```python
def convene_round_robin(elders: list, max_rounds: int, topic: str):
    """Simplified sketch of Tribunal.convene()'s core loop -- see
    code/elders_tribunal/orchestrator.py for the full, real implementation
    with the DebateResult dataclass and moderator call wired in.
    """
    history: list = []
    stop_reason = "max_rounds"
    round_verdicts: dict = {}

    for round_no in range(1, max_rounds + 1):
        round_verdicts = {}
        for elder in elders:
            text = elder.respond(topic, list(history))
            history.append((elder.name, round_no, text))
            round_verdicts[elder.name] = extract_verdict(text)

        cast = [v for v in round_verdicts.values() if v is not None]
        everyone_voted = len(cast) == len(elders)
        if everyone_voted and len(set(cast)) == 1:
            stop_reason = "unanimous"
            break

    return history, round_verdicts, stop_reason
```

Two details worth being deliberate about, both visible in that loop:

- **Every elder sees the growing history, including same-round statements from elders ahead of it.**
  `list(history)` is passed by value at each call — a defensive copy, so no elder implementation can
  mutate the shared transcript out from under the orchestrator (Python lists are mutable and passed
  by reference; copying here is the equivalent of a Java method returning
  `Collections.unmodifiableList(...)`, done manually since dataclasses don't do it for you).
- **The unanimous check runs after every round, not just the last one.** That is what makes early
  stopping possible at all — Section 3.3's test asserts an elder is *never called again* once the
  panel agrees.

Full file, including the real `Tribunal` class, `DebateResult` dataclass, and `format_report()`
helper: [code/elders_tribunal/orchestrator.py](code/elders_tribunal/orchestrator.py).

### 3.3 Testing with scripted fakes — no key, no network, fully deterministic

`test_orchestrator.py` uses **only `unittest`**, Python's standard-library test framework — no
`pytest`, matching this project's `.venv-agent`, which does not have `pytest` installed. Every elder
in the test suite is a **scripted fake**: it returns the next line from a fixed list every time
`respond()` is called, exactly the same test-double pattern as a Java `FakeRepository implements
Repository` used in place of a real JDBC-backed one — the fake satisfies the `Elder` interface, so
`Tribunal` cannot tell it apart from a real, LLM-backed elder:

```python
from elder import Elder


class ScriptedElder(Elder):
    """Returns the next line from a fixed script each call; repeats the last
    line if called more times than the script is long. `self.calls` lets a
    test assert an elder was (or was NOT) called again.
    """

    def __init__(self, name: str, script: list) -> None:
        super().__init__(name)
        self._script = script
        self.calls = 0

    def respond(self, topic: str, history: list) -> str:
        text = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return text
```

The early-stop test is the one that actually proves the "unanimous" branch works, not just that it
compiles — it asserts a second round's elder was **never invoked**:

```python
import unittest

from orchestrator import Tribunal


class TestEarlyStop(unittest.TestCase):
    def test_early_stop_on_unanimous_verdict_skips_remaining_rounds(self) -> None:
        elder_a = ScriptedElder("Elder-A", ["Agree. VERDICT: YES", "SHOULD NOT BE CALLED"])
        elder_b = ScriptedElder("Elder-B", ["Agree too. VERDICT: YES", "SHOULD NOT BE CALLED"])
        tribunal = Tribunal([elder_a, elder_b], ScriptedModerator(), max_rounds=5)

        result = tribunal.convene("Should we adopt trunk-based development?")

        self.assertEqual(result.stop_reason, "unanimous")
        self.assertEqual(result.rounds_run, 1)
        self.assertEqual(elder_a.calls, 1)  # round 2 never happened
        self.assertEqual(elder_b.calls, 1)
```

The full suite (10 tests) also covers round-robin ordering and history growth, max-rounds
termination, majority voting, deadlock detection, missing-verdict-as-abstention, case-insensitive
verdict extraction, and the constructor guards (`ValueError` on fewer than two elders or
`max_rounds < 1`). Full file:
[code/elders_tribunal/test_orchestrator.py](code/elders_tribunal/test_orchestrator.py).

### Real, captured test run

```text
.venv-agent/Scripts/python.exe -m unittest test_orchestrator.py -v
```

```text
test_deadlock_when_no_verdict_holds_a_strict_majority ... ok
test_majority_vote_wins_without_unanimity ... ok
test_missing_verdict_is_treated_as_abstention_not_an_error ... ok
test_verdict_extraction_is_case_insensitive_and_ignores_surrounding_text ... ok
test_requires_at_least_one_round ... ok
test_requires_at_least_two_elders ... ok
test_early_stop_on_unanimous_verdict_skips_remaining_rounds ... ok
test_max_rounds_reached_without_unanimity ... ok
test_round_robin_order_and_history_growth ... ok
test_second_speaker_sees_first_speakers_statement_same_round ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.001s

OK
```

Full captured log: [artefacts/test_orchestrator_run_log.txt](artefacts/test_orchestrator_run_log.txt).
Ten for ten, zero network calls, zero keys. This is the entire proof this chapter offers for LO2: the
protocol from Section 2 is not just described, it is exercised, and the exercise is repeatable by
anyone who clones this repository.

## 4. Wiring real, different providers — key-gated, clearly separate (LO3)

Everything in Section 3 proves the *orchestrator* works with zero LLM involvement. The other half of
this chapter's promise is that the exact same `Elder` interface can be implemented against a real
model — and that you can put genuinely different providers on the same panel, not three instances of
the same model with different names.

`providers.py` implements three concrete elders, one per provider, using package versions and model
IDs verified live and recorded in
[research/NOTE-AGENT-4-provider-sdks.md](../../research/NOTE-AGENT-4-provider-sdks.md) (checked
2026-09-02):

| Provider | Package (verified version) | Env var | Model used here |
|---|---|---|---|
| Anthropic | `anthropic==1.3.0` | `ANTHROPIC_API_KEY` | `claude-opus-5` (latest Claude model, released 2026-07-24, per NOTE-AGENT-4) |
| OpenAI | `openai==3.7.0` | `OPENAI_API_KEY` | `gpt-4o` |
| Google | `google-genai==2.22.0` | `GOOGLE_API_KEY` | `gemini-2.5-flash` |

None of these three packages is installed in this project's shared `.venv-agent` (confirmed while
writing this chapter), and this chapter never installs them — `providers.py` is reference code you
run in your own environment with your own keys. What makes that safe to leave uninstalled is that
every provider SDK import is **lazy**, deferred to inside each elder's `__init__`, and each
`__init__` checks its API key *before* attempting that import:

```python
import os


class AnthropicElder:
    """Sketch of the real class -- see code/elders_tribunal/providers.py for
    the full version implementing the Elder interface from elder.py.
    """

    def __init__(self, name: str = "Anthropic-Elder", model: str = "claude-opus-5") -> None:
        self.name = name
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. AnthropicElder makes a real, "
                "billed API call and refuses to construct without a key."
            )
        from anthropic import Anthropic  # lazy: only imported once a key exists

        self._model = model
        self._client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
```

That ordering — check the key, raise if missing, *then* `import anthropic` — is why
`python -m py_compile providers.py` succeeds in an environment with none of the three SDKs
installed, and why merely `import providers` never fails either: nothing in the module executes an
`import anthropic` / `import openai` / `from google import genai` statement at module load time, only
inside a constructor that already refused to run without its key. The minimal chat-call shape for
each provider is exactly what NOTE-AGENT-4 verified against each SDK's official docs:

```python
def anthropic_minimal_call_shape(client, model: str, prompt: str) -> str:
    """The exact call shape verified in NOTE-AGENT-4 against
    platform.claude.com/docs/en/build-with-claude/working-with-messages
    (checked 2026-09-02). No temperature/top_p/top_k -- unsupported on
    Claude 4.7+ per that same note; use structured prompting instead.
    """
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
```

`OpenAIElder` and `GoogleElder` in the full file follow the identical pattern — check key, lazy
import, one `_render_prompt()` helper shared by all three so they differ only in *which SDK they
call*, never in what they're asked. `providers.py` also defines
`SynthesizingAnthropicModerator`, a real `Moderator` implementation that asks Claude to write the
consensus report instead of formatting it with string templates — same key-gated discipline, same
lazy import. Full file: [code/elders_tribunal/providers.py](code/elders_tribunal/providers.py).

### What this chapter does *not* do: fabricate a "live" transcript

`providers.py` has an `if __name__ == "__main__":` block, mirroring the pattern the MCP chapter used
for its own key-gated LLM client
([llm_client.py](code/mcp_db/llm_client.py) /
[mcp-database-query-layer.md Section 5](mcp-database-query-layer.md#5-wiring-an-llm-client--key-gated-separate-from-the-runnable-path)):
with no keys set, it prints what's missing and exits — no SDK import, no network call:

```text
.venv-agent/Scripts/python.exe providers.py
```

```text
Missing ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY -- skipping the live, multi-provider tribunal. Set every key above (in .env, never committed) and install the matching SDKs (anthropic==1.3.0, openai==3.7.0, google-genai==2.22.0) to run a real cross-provider debate. This script makes no network call without them.
```

(Full transcript: [artefacts/providers_no_key_transcript.txt](artefacts/providers_no_key_transcript.txt) —
captured for real, on 2026-09-03, from this exact repository.)

This chapter does **not** paste in a hand-typed "Claude says X, GPT-5 says Y, Gemini says Z"
transcript here. Doing that would mean typing out what three live models would say without any of
them having said it — precisely the kind of ungrounded claim this project's rules forbid (see
`docs/style-guide.md`'s grounding conventions). What you have instead is stronger: the real no-key
path above, the full, real code for all three elders, and Section 5's fully real mock-backed
consensus — the exact same orchestrator, moderator contract, and report format `providers.py` would
produce, just driven by scripted fakes instead of three paid API calls. Point `providers.py` at your
own keys to see the genuine live version on your own machine.

### Why not Google ADK for the orchestration layer?

NOTE-AGENT-4 also checked whether Google's Agent Development Kit (ADK, `google-adk==2.8.0`) offers a
native multi-agent primitive worth using here instead of a hand-rolled orchestrator. ADK does provide
real multi-agent building blocks — `SequentialAgent`, `ParallelAgent`, `LoopAgent` — but they are part
of a full agentic framework tied to Google's own runtime and credentials, not a lightweight
turn-taking primitive you could drop under an already-defined `Elder` interface. `google-genai`
itself (the SDK used for `GoogleElder` above) has no multi-agent orchestration at all — it is a
single-model chat/tool-use client, confirmed in NOTE-AGENT-4. For a chapter whose whole point is a
provider-agnostic interface that Anthropic, OpenAI, and Google implementations all satisfy equally,
committing the orchestration layer itself to one vendor's framework would work against the lesson —
so this chapter keeps the hand-rolled `Tribunal`, exactly as NOTE-AGENT-4 recommends.

## 5. Consensus — voting vs. moderator synthesis, and a real captured debate (LO4)

Section 2 already named the two techniques the orchestrator always runs, together, at the end of
every debate:

- **Voting** (`_tally()`, Section 3.2) — a mechanical count of the final round's `VERDICT:` tags.
  Deterministic, cheap, and honest about ties: a verdict only wins with a *strict* majority
  (`> len(elders) / 2`); anything else — including an exact split — reports `winning_verdict = None`
  and `deadlock = True`. No tie-breaking rule quietly picks a winner.
- **Moderator synthesis** (`moderator.synthesize(topic, history, votes)`) — always runs too, and is
  handed the *same* vote tally the numerical count produced, plus the full transcript. Its job is to
  turn a number into a sentence a human can act on: "unanimous", "majority, here's the split", or
  explicitly "no consensus, here is why" — never to override what the vote already said.

`run.py` demonstrates both outcomes with a `SynthesizingModerator` that formats the vote tally into
prose without calling an LLM (a real deployment would swap this for `providers.py`'s
`SynthesizingAnthropicModerator`, which asks Claude to write the same report — the `Moderator`
contract does not change either way):

```python
class SynthesizingModerator:
    """Sketch -- the real class implements orchestrator's Moderator interface;
    see code/elders_tribunal/run.py for the full version.
    """

    def synthesize(self, topic: str, history: list, votes: dict) -> str:
        cast = {name: v for name, v in votes.items() if v is not None}
        if cast and len(set(cast.values())) == 1:
            verdict = next(iter(cast.values()))
            return (
                f"Consensus: the tribunal unanimously reports '{verdict}' on "
                f"'{topic}' ({len(cast)}/{len(votes)} elders cast a verdict)."
            )
        tally: dict = {}
        for verdict in cast.values():
            tally[verdict] = tally.get(verdict, 0) + 1
        ordered = sorted(tally.items(), key=lambda item: -item[1])
        breakdown = ", ".join(f"{v}={n}" for v, n in ordered)
        if ordered and ordered[0][1] > len(votes) / 2:
            return f"Consensus by majority: '{ordered[0][0]}' on '{topic}' ({breakdown})."
        return (
            f"No consensus on '{topic}' -- the panel is split ({breakdown}). "
            "Recommending the team gather more evidence or escalate to a human "
            "decision-maker rather than forcing a false majority."
        )
```

That last branch is the deadlock-handling policy this chapter picked, stated as code: **when the
panel is genuinely split, say so, and recommend escalation — never fabricate a majority to look
decisive.** A different, equally valid policy would be "fall back to a designated tie-breaking
elder" or "run one more round with a smaller, targeted question" — the point is that *some* explicit
policy runs here, rather than the orchestrator silently defaulting to whichever verdict happened to
be extracted first.

### Real, captured run — mock-backed, deterministic, no key

`run.py` convenes two debates, both with scripted `PersonaElder`s (same script-driven fake pattern as
Section 3.3's `ScriptedElder`, with persona-flavoured text instead of test-focused placeholder text).
Nothing here is hand-typed into this chapter — this is `run.py`'s actual stdout, captured on
2026-09-03:

```text
.venv-agent/Scripts/python.exe run.py
```

```text
========================================================================
DEBATE 1 -- three elders, full round cap, majority (not unanimous)
========================================================================
Topic: Should this 4-engineer greenfield team start with microservices?
Rounds run: 3 (stop reason: max_rounds)

--- Transcript ---
[Round 1] Pragmatist-Elder: Ship the monolith first -- microservices add operational overhead this 4-person team hasn't earned yet. VERDICT: NO
[Round 1] Architect-Elder: Start with clear module boundaries inside one deployable; split into services only once a boundary proves it needs to scale independently. VERDICT: NO
[Round 1] Scale-Elder: If the projected growth is real, retrofitting service boundaries later costs more than designing them in from day one. VERDICT: YES
[Round 2] Pragmatist-Elder: Even after hearing the scalability case, the coordination cost of running N services outweighs the benefit for a team this size today. VERDICT: NO
[Round 2] Architect-Elder: Agreed with Pragmatist-Elder -- premature service boundaries are harder to undo than a well-modularised monolith. VERDICT: NO
[Round 2] Scale-Elder: I hear the team-size argument, but I still think the retrofit cost is being understated. Sticking with my position for now. VERDICT: YES
[Round 3] Pragmatist-Elder: Position unchanged: revisit this once the team has grown or traffic data justifies the split. VERDICT: NO
[Round 3] Architect-Elder: Holding this position through the final round. VERDICT: NO
[Round 3] Scale-Elder: Still YES, though I'll concede this is now a minority view on this panel. VERDICT: YES

--- Votes (final round) ---
  Pragmatist-Elder: NO
  Architect-Elder: NO
  Scale-Elder: YES

Unanimous: False   Deadlock: False   Winning verdict: NO

--- Consensus (moderator) ---
Consensus by majority: 'NO' on 'Should this 4-engineer greenfield team start with microservices?' (NO=2, YES=1, out of 3 elders).

========================================================================
DEBATE 2 -- three-way split: deadlock, no forced consensus
========================================================================
Topic: Should we rewrite the auth service in Rust this quarter?
Rounds run: 1 (stop reason: max_rounds)

--- Transcript ---
[Round 1] Elder-A: Rewrite the auth service in Rust this quarter. VERDICT: YES
[Round 1] Elder-B: Keep it in Python -- the risk isn't worth it right now. VERDICT: NO
[Round 1] Elder-C: Neither yet -- prototype both for two weeks before deciding. VERDICT: UNCERTAIN

--- Votes (final round) ---
  Elder-A: YES
  Elder-B: NO
  Elder-C: UNCERTAIN

Unanimous: False   Deadlock: True   Winning verdict: None

--- Consensus (moderator) ---
No consensus on 'Should we rewrite the auth service in Rust this quarter?' -- the panel is split (YES=1, NO=1, UNCERTAIN=1). Recommending the team gather more evidence or escalate to a human decision-maker rather than forcing a false majority.
```

Full transcript: [artefacts/mock_debate_transcript.txt](artefacts/mock_debate_transcript.txt). Two
outcomes, both real:

- **Debate 1** never reaches unanimity (Scale-Elder holds its `YES` through all three rounds) but
  *does* have a strict majority (2 of 3 `NO`) once the round cap is hit — `winning_verdict = "NO"`,
  `deadlock = False`, and the moderator's report names the exact 2-1 split rather than just saying
  "NO".
- **Debate 2** is a genuine three-way split with no verdict holding a strict majority —
  `winning_verdict = None`, `deadlock = True`, and the moderator's report says so explicitly and
  recommends escalation instead of quietly picking `YES` (which was, after all, the first verdict
  extracted).

Full file: [code/elders_tribunal/run.py](code/elders_tribunal/run.py).

## 6. Pitfalls

### 6.1 Cost blowup

Every round is `len(elders)` API calls; every extra elder or extra round multiplies the bill linearly
— a 3-elder, 3-round debate that never stops early is 9 calls for what a single-agent design would
answer in one. `max_rounds` and the unanimous-early-stop check (Section 2) are not just protocol
niceties, they are the only two things standing between this pattern and an unbounded per-request
cost. Before adding a fourth elder or raising `max_rounds`, ask whether the question actually needs
that much deliberation — most don't.

### 6.2 Echo chambers

Nothing stops three elders backed by the *same* underlying model (three `AnthropicElder` instances
with different names, say) from producing what looks like "independent consensus" but is really one
model agreeing with itself three times, because it was never actually a different opinion to begin
with. The "different failure modes cancel out" argument from Section 1 only holds if the panel is
genuinely heterogeneous — different providers (Section 4), or at minimum different prompts/roles on
the same provider. A tribunal of clones reaching "unanimous" consensus fast should read as a red
flag, not a good result.

### 6.3 Inter-agent prompt injection

Every elder's `respond()` call includes the full transcript of what every *other* elder has said —
that is the entire mechanism that lets elder 2 rebut elder 1. It is also an attack surface: if any
one elder is compromised (a malicious system prompt, a poisoned model, or in the real-provider case a
model that decides to role-play adversarially), it can write text specifically crafted to manipulate
how a *later* elder — or the moderator — interprets the debate, the same class of problem as
untrusted input reaching a downstream service that trusts it. This chapter's mock elders can't do
this (they read a fixed script, never the history they're handed), but a real provider-backed elder
reads every other elder's raw text as part of its own prompt with no sanitisation in `providers.py`.
Treat every other elder's statement as untrusted input to the next one, the same discipline this
project applies to any text an LLM produces that another system later consumes.

### 6.4 Non-determinism

`run.py`'s output above is reproducible byte-for-byte because every elder is a scripted fake —
running it twice gives you the identical transcript, which is why this chapter could capture it and
paste it in verbatim. A `providers.py`-backed debate has no such guarantee: each provider's own
sampling introduces run-to-run variation (NOTE-AGENT-4 notes Claude models 4.7+ don't even expose
`temperature`/`top_p`/`top_k` as knobs to reduce it), so the same topic run twice against real
providers can produce different transcripts, different verdicts, and — in a close vote — a different
winner. If you need a deterministic tribunal for testing purposes, use scripted `Elder`
implementations, exactly as `test_orchestrator.py` and `run.py` do; if you need a deterministic
*real* debate, you don't currently have one available from any of the three providers in this
chapter's scope.

## 7. Recap & what's next

- **Multi-agent debate is a protocol, not a framework choice**: fixed roles (elders + one
  moderator), round-robin turn order, and two independent stop conditions (unanimous early stop, hard
  round cap) — simple enough to trace by hand, which is exactly why SPEC-AGENT-5 scoped out
  "elaborate agent frameworks" (LO1, Section 2).
- **`Elder` is one abstract method** (`respond(topic, history) -> str`), with zero third-party
  imports — the same dependency-inversion shape as coding against a Java interface instead of a
  concrete implementation. `Tribunal` depends only on that interface and its `Moderator` sibling, so
  it is fully unit-testable with scripted fakes: 10/10 tests pass with zero keys and zero network
  calls (LO2, Section 3).
- **Real providers wire in behind the exact same interface**, key-gated with lazy SDK imports so
  `providers.py` compiles and imports cleanly in an environment with none of the three SDKs
  installed — versions and model IDs (`anthropic==1.3.0`/`claude-opus-5`,
  `openai==3.7.0`/`gpt-4o`, `google-genai==2.22.0`/`gemini-2.5-flash`) verified live in
  [NOTE-AGENT-4](../../research/NOTE-AGENT-4-provider-sdks.md). No fabricated "live" transcript is
  presented anywhere in this chapter — only the real no-key path, captured for real (LO3, Section 4).
- **Consensus is both a number and a sentence**: `_tally()`'s strict-majority vote count, and the
  moderator's `synthesize()` report, always run together — Debate 1's real 2-1 majority and Debate
  2's real three-way deadlock (captured in
  [artefacts/mock_debate_transcript.txt](artefacts/mock_debate_transcript.txt)) show both outcomes
  from one honest policy: report a majority when one exists, report the split and recommend
  escalation when it doesn't (LO4, Section 5).
- **Pitfalls that are specific to multi-agent systems**, not single-agent ones: cost scales with
  `elders x rounds`; a panel of clones is an echo chamber wearing consensus's clothes; every elder's
  prompt includes every other elder's raw, unsanitised text; and only the scripted core is
  deterministic — a real, provider-backed debate is not (Section 6).

The next Agentic Engineering chapter, cloud deployment
([SPEC-AGENT-6](../../specs/SPEC-AGENT-6-cloud-deployment.md)), takes an agentic application —
this tribunal, or any of the earlier worked examples — and moves it off a local machine and into a
managed cloud environment, where the key-gating discipline this chapter and
[the MCP chapter](mcp-database-query-layer.md) both practised (secrets in env vars, never in code)
stops being a nice habit and becomes the actual security boundary.
