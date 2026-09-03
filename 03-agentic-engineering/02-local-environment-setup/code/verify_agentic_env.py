"""Verify the Agentic Engineering local environment.

Two separate stacks, on purpose (the chapter explains why):

1. The shared `.venv-agent` stack -- fastapi, fastmcp, pgvector (Python client), psycopg,
   pydantic. This script's `check_agent_stack()` imports each and prints its installed
   version. Run this half with `.venv-agent`'s interpreter.
2. `google-adk`, which this project keeps in its OWN virtualenv, `.venv-adk` (see the
   chapter, Section 2.3, for the isolation rationale -- it is NOT about a live version
   conflict; empirically, `google-adk==2.8.0` resolves cleanly against
   `fastapi==0.141.1`/`starlette==1.6.0`, confirmed by `pip install --dry-run` and
   `pip check` against this project's real `.venv-agent`, both logged in the chapter).
   `check_adk()` handles ADK being ABSENT gracefully -- if you run this script with
   `.venv-agent`'s interpreter (where ADK is deliberately not installed), it reports that
   plainly instead of raising ImportError and killing the whole check.

Run against .venv-agent (fastapi/fastmcp/pgvector/psycopg -- ADK reported absent, by design):
    .venv-agent\\Scripts\\python.exe "Agentic Engineering/Local Environment Setup/code/verify_agentic_env.py"

Run against .venv-adk (only google-adk is checked; the other imports are reported absent there
by design too -- it is a dedicated, minimal environment):
    .venv-adk\\Scripts\\python.exe "Agentic Engineering/Local Environment Setup/code/verify_agentic_env.py"

Expects (pinned in this chapter; see local-environment-setup.md and
research/NOTE-AGENT-1-stack.md, checked 2026-09-02):
    fastapi==0.141.1  fastmcp==4.0.1  pgvector==0.5.0  psycopg==3.3.5
    google-adk==2.8.0 (separate venv)

No LLM API key is required for anything this script checks -- it only imports libraries and
reads their reported version. The LLM-call path (Section 4 of the chapter) is separately
marked key-gated and is not exercised here.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version


def check_agent_stack() -> None:
    """Imports and reports the shared .venv-agent stack. Each import is independent --
    one missing package is reported and skipped, not a fatal crash, so this script gives
    a full picture even against a partially-installed environment."""
    packages = ["fastapi", "fastmcp", "pgvector", "psycopg", "pydantic"]
    print("-- shared .venv-agent stack --")
    for name in packages:
        try:
            __import__(name)
            print(f"{name:<12} {pkg_version(name)}")
        except ImportError:
            print(f"{name:<12} NOT INSTALLED in this interpreter")
        except PackageNotFoundError:
            print(f"{name:<12} imported, but no distribution metadata found")


def check_adk() -> None:
    """Imports google-adk if present; reports 'not installed here' if not, instead of
    raising -- this is the graceful-absence behaviour the chapter calls for, since this
    project deliberately does NOT install google-adk into .venv-agent."""
    print("-- google-adk (expected in a SEPARATE .venv-adk, not here) --")
    try:
        from google.adk.agents import Agent  # noqa: F401

        print(f"google-adk     {pkg_version('google-adk')}  (import OK: google.adk.agents.Agent)")
    except ImportError:
        print("google-adk     NOT INSTALLED in this interpreter (expected if this is .venv-agent)")


def main() -> None:
    print(f"Python:        {sys.version.split()[0]}")
    print(f"Executable:    {sys.executable}")
    print()
    check_agent_stack()
    print()
    check_adk()


if __name__ == "__main__":
    main()
