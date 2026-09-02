"""KEY-GATED REFERENCE: real, DIFFERENT LLM providers wired behind the same
`Elder` interface from elder.py (LO3).

This file is intentionally NEVER imported by test_orchestrator.py or run.py --
those two stay 100% offline, no key, no provider SDK required, per SPEC-AGENT-5.
Everything here requires network access and a billed API key, so it is kept
completely separate, the same way the MCP chapter kept `llm_client.py` separate
from its no-key `test_client.py`
(../mcp_db/llm_client.py, Agentic Engineering/Worked Examples/mcp-database-query-layer.md
Section 5).

Every provider SDK import below happens LAZILY, inside each elder's
`__init__`, specifically so that `import providers` and
`python -m py_compile providers.py` never require anthropic / openai /
google-genai to be installed -- only actually CONSTRUCTING one of these
classes does. None of the three packages is installed in this project's shared
`.venv-agent` (confirmed while writing this chapter, 2026-09-03); this module
compiles cleanly there anyway, and only fails -- loudly, with a clear message
-- if you try to instantiate an elder without its key and package.

Package versions, model ids, and the minimal chat-call shape below are all
verified live on PyPI / the official docs and grounded in
[research/NOTE-AGENT-4-provider-sdks.md](../../../research/NOTE-AGENT-4-provider-sdks.md)
(checked 2026-09-02):

    anthropic==1.3.0     ANTHROPIC_API_KEY   model: claude-opus-5
    openai==3.7.0         OPENAI_API_KEY      model: gpt-4o (or gpt-5, once available)
    google-genai==2.22.0  GOOGLE_API_KEY      model: gemini-2.5-flash

NOTE-AGENT-4 also confirms Claude models 4.7+ do not support sampling
parameters (temperature / top_p / top_k) -- none are passed below, on purpose.
"""
from __future__ import annotations

import os

from elder import Elder, Moderator, Statement


def _render_prompt(topic: str, history: list[Statement]) -> str:
    """Build the one prompt string every provider elder below sends -- shared
    so the three implementations differ only in which SDK they call, not in
    what they say.
    """
    lines = [f"Topic under debate: {topic}", ""]
    if not history:
        lines.append("You are opening the debate. No one has spoken yet.")
    else:
        for statement in history:
            lines.append(f"[Round {statement.round_no}] {statement.speaker}: {statement.text}")
    lines.append("")
    lines.append(
        "Give your position in 2-4 sentences, then end your reply with a line "
        "reading exactly: VERDICT: <YES|NO|UNCERTAIN>"
    )
    return "\n".join(lines)


class AnthropicElder(Elder):
    """An elder backed by Anthropic's Claude, via the `anthropic` SDK (1.3.0).

    REQUIRES: ``pip install anthropic==1.3.0`` and the ``ANTHROPIC_API_KEY``
    env var. Raises immediately, in ``__init__``, if the key is missing --
    this class never makes a network call "by accident".
    """

    def __init__(self, name: str = "Anthropic-Elder", model: str = "claude-opus-5") -> None:
        super().__init__(name)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. AnthropicElder makes a real, "
                "billed API call and refuses to construct without a key."
            )
        from anthropic import Anthropic  # lazy: only imported once a key exists

        self._model = model
        self._client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment itself

    def respond(self, topic: str, history: list[Statement]) -> str:
        # Minimal chat-call shape verified in NOTE-AGENT-4 against the
        # official docs (platform.claude.com/docs/en/build-with-claude/
        # working-with-messages, checked 2026-09-02). No temperature/top_p/
        # top_k -- unsupported on Claude 4.7+ per that same note.
        message = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": _render_prompt(topic, history)}],
        )
        return message.content[0].text


class OpenAIElder(Elder):
    """An elder backed by OpenAI, via the `openai` SDK (3.7.0).

    REQUIRES: ``pip install openai==3.7.0`` and the ``OPENAI_API_KEY`` env var.
    """

    def __init__(self, name: str = "OpenAI-Elder", model: str = "gpt-4o") -> None:
        super().__init__(name)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. OpenAIElder makes a real, billed "
                "API call and refuses to construct without a key."
            )
        from openai import OpenAI  # lazy: only imported once a key exists

        self._model = model
        self._client = OpenAI()  # reads OPENAI_API_KEY from the environment itself

    def respond(self, topic: str, history: list[Statement]) -> str:
        # Minimal chat-call shape verified in NOTE-AGENT-4 against the
        # official OpenAI Python API reference (checked 2026-09-02). "gpt-4o"
        # is the documented example model id; swap for "gpt-5" once you have
        # verified that id is live on your account.
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": _render_prompt(topic, history)}],
        )
        return completion.choices[0].message.content


class GoogleElder(Elder):
    """An elder backed by Google Gemini, via the `google-genai` SDK (2.22.0).

    REQUIRES: ``pip install google-genai==2.22.0`` and the ``GOOGLE_API_KEY``
    env var. Package name is ``google-genai`` -- NOT the deprecated
    ``google-generativeai`` (NOTE-AGENT-4, caveat 4).
    """

    def __init__(self, name: str = "Google-Elder", model: str = "gemini-2.5-flash") -> None:
        super().__init__(name)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. GoogleElder makes a real, billed "
                "API call and refuses to construct without a key."
            )
        from google import genai  # lazy: only imported once a key exists

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def respond(self, topic: str, history: list[Statement]) -> str:
        # google-genai's chat session keeps its own turn history internally;
        # this elder still builds one self-contained prompt per call (like the
        # other two) so all three elders share the exact same `_render_prompt`
        # contract and the orchestrator never needs to know the difference.
        chat = self._client.chats.create(model=self._model)
        response = chat.send_message(_render_prompt(topic, history))
        return response.text


class SynthesizingAnthropicModerator(Moderator):
    """A real moderator: asks Claude to read the transcript and vote tally and
    write the consensus report, instead of the string-formatting
    `SynthesizingModerator` in run.py. Same KEY-GATED discipline as the elders
    above -- lazy import, refuses to construct without a key.
    """

    def __init__(self, name: str = "Anthropic-Moderator", model: str = "claude-opus-5") -> None:
        super().__init__(name)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. SynthesizingAnthropicModerator "
                "refuses to construct without a key."
            )
        from anthropic import Anthropic

        self._model = model
        self._client = Anthropic()

    def synthesize(
        self, topic: str, history: list[Statement], votes: dict[str, str | None]
    ) -> str:
        transcript_text = "\n".join(
            f"[Round {s.round_no}] {s.speaker}: {s.text}" for s in history
        )
        prompt = (
            f"Topic: {topic}\n\nFull debate transcript:\n{transcript_text}\n\n"
            f"Final-round votes: {votes}\n\n"
            "Write a short consensus report: state whether the panel reached "
            "consensus, and if not, summarise the disagreement honestly -- do "
            "not invent a winner that the votes do not support."
        )
        message = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


def _missing_keys() -> list[str]:
    return [
        var
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
        if not os.environ.get(var)
    ]


if __name__ == "__main__":
    # Mirrors the mcp_db chapter's llm_client.py pattern: with no keys set,
    # this prints a message and exits -- no import of any provider SDK is even
    # attempted, and no network call is made.
    missing = _missing_keys()
    if missing:
        print(
            "Missing " + ", ".join(missing) + " -- skipping the live, "
            "multi-provider tribunal. Set every key above (in .env, never "
            "committed) and install the matching SDKs "
            "(anthropic==1.3.0, openai==3.7.0, google-genai==2.22.0) to run a "
            "real cross-provider debate. This script makes no network call "
            "without them."
        )
    else:
        from orchestrator import Tribunal, format_report

        panel: list[Elder] = [AnthropicElder(), OpenAIElder(), GoogleElder()]
        moderator = SynthesizingAnthropicModerator()
        tribunal = Tribunal(elders=panel, moderator=moderator, max_rounds=3)
        result = tribunal.convene(
            "Should this team adopt a monorepo for its next three services?"
        )
        print(format_report(result))
