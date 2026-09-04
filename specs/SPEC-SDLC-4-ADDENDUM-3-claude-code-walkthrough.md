# SPEC-SDLC-4-ADDENDUM-3: walk a newcomer through setting up Claude Code and vibe-engineering the store himself

**Status:** approved
**Subject:** AI-assisted-sdlc
**Section:** Worked Examples (amends `03-rails-estore-sdlc.md` and `code/rails-estore/README.md`)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Sequencing:** land AFTER the docker-compose addendum (SPEC-SDLC-4-ADDENDUM-2) is committed — this
builds on that README (primer + quickstart + screenshot).

## Why
The real intent of this whole example is not "here is a finished Rails store to read." It is: **a
newcomer sets up Claude Code and builds the store himself**, steering the agents under the governance
scaffold — "vibe engineering" with guardrails. So the chapter and (especially) the standalone README
must become a **hands-on, do-it-yourself walkthrough**: install Claude Code, open this scaffold, and
drive the implementer/reviewer/seo/frontend-qa agents to build login → checkout → catalog, watching the
gates keep it safe, and finally run it with `docker compose up`. The committed `rails-estore/` code is
reframed as the **reference destination** ("where you'll end up"), not the point.

Audience: someone new to Claude Code AND new to this workflow (he's already getting the Docker primer
and the macOS setup from ADDENDUM-2). Assume competence as a developer, zero Claude Code experience.

## What to add / reframe
### 1. "Set up Claude Code" (README section, grounded)
A concise, correct getting-started: what Claude Code is in one paragraph (an agentic coding tool in the
terminal that reads this project's `CLAUDE.md` rules, dispatches the `.claude/agents/` sub-agents, and
runs the `.claude/hooks/` gates on every edit); **install** it (the current official method); **sign
in** (Anthropic account / the documented auth); open the project (`cd code/rails-estore && claude`);
and the idea of **permission modes** (why it asks before running shell / editing — and that this
project's `guard.sh` adds its own hard blocks on top). Everything grounded to the official Claude Code
docs, not memory.

### 2. "Vibe-engineer the store, step by step" (the heart — README + a chapter section)
A numbered walkthrough the friend actually follows, building the app himself through the governed loop.
For each feature, the SAME rhythm (this is the "vibe": you steer, the agents implement under guardrails,
you review and decide, the gates keep it safe):
1. **Read the feature spec** (`docs/features/FEATURE-1-user-login.md`) — the acceptance criteria are the
   contract.
2. **Ask Claude Code to implement it** — a good starting prompt; it dispatches the **implementer**
   sub-agent, which writes the failing RSpec example first, then the code.
3. **Watch the gates fire** — `verify.sh` runs RSpec/RuboCop/Brakeman on each edit; `guard.sh` blocks
   dangerous shell and secrets. Show what a red gate looks like and how you get to green.
4. **Get an independent review** — dispatch the **reviewer** (and for the catalog, **seo-optimizer** +
   **frontend-qa**); see it request a change (tie to the real IDOR / a11y / SEO catches already in the
   chapter) and iterate.
5. **Repeat** for FEATURE-2 (checkout) and FEATURE-3 (catalog + SEO/a11y).
6. **Run what you built** — `docker compose up`, open `http://localhost:3000` (link to the screenshot).
Include example prompts the friend can paste, and a short "how to steer well" aside (be specific, point
at the spec, let the gates and reviewers do the catching, review every diff yourself). Honest notes:
model outputs vary run to run; you stay in the loop; the governance is what makes hands-off stretches
safe.

### 3. Tie it together
Update the chapter's framing/recap and the README intro so the through-line is explicit: *you* set up
Claude Code and build this, the scaffold governs it, the gates + specialist agents catch what you'd
miss, and docker-compose runs the result. Keep every existing section (Docker primer, quickstart,
screenshot, macOS setup, the governance story) — this adds the connective "do it yourself" spine.

## Grounding (Haiku — do BEFORE writing)
- [ ] The current, official way to **install Claude Code** (the command / installer) and **sign in**,
      from the official docs (docs.claude.com / docs.anthropic.com Claude Code) — with the date checked.
      Do not assert the install command from memory.
- [ ] How Claude Code uses **`CLAUDE.md`**, **`.claude/agents/` sub-agents** (how they're invoked —
      automatically and/or via `/agents`), **`.claude/hooks/`** (PreToolUse/PostToolUse via
      `settings.json`), and **permission modes** — cite the official sub-agents / hooks / settings docs.
- [ ] Confirm the `claude` CLI basics a newcomer needs (starting a session in a directory, that it
      reads the project's `CLAUDE.md`, approving actions) — cite.

## Acceptance criteria
- [ ] AC1 — a grounded "Set up Claude Code" section (install + sign-in + open project + permission
      modes) is in the README → evidence: the section + citations.
- [ ] AC2 — a "vibe-engineer it yourself" numbered walkthrough (in README + a chapter section) drives
      building login/checkout/catalog through the governed loop, with example prompts and the
      steer/implement/gate/review rhythm, ending at `docker compose up` → the screenshot.
- [ ] AC3 — the chapter + README framing reposition `rails-estore/` as the destination the reader
      builds, not a finished artefact to skim; existing sections preserved.
- [ ] AC4 — every Claude Code install/command/behaviour claim grounded (NOTE ids / inline citation +
      date); honest that outputs vary and the human stays in the loop.
- [ ] AC5 — renders on GitHub (`check_markdown_render.py` on the chapter + README); no key-shaped
      strings; coherence: `04-ai-assisted-sdlc/README.md` and the `docs/curriculum.md` SDLC-4 bullet
      (architect) note the DIY Claude-Code walkthrough.

## Gates
Exit: ACs satisfied; chapter + README render; links resolve; fresh-Sonnet review; architect merge.
(See `docs/definition-of-done.md`.)
