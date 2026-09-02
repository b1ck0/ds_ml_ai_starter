# NOTE-SDLC-2: Claude Code — Installation, Surfaces, Hooks, Agents, Skills, MCP, Rules, Permissions (2026)

**Answer:** Claude Code installs via native installer (auto-update, no runtime needed); authenticates via OAuth (local token) or API key (`claude login` or env var); supports CLI (terminal), IDE (VS Code / JetBrains), desktop app, and web; hooks fire on lifecycle events (SessionStart, PreToolUse, PostToolUse, UserPromptSubmit, Stop, etc.) with settings.json schema supporting command/http/mcp_tool/prompt/agent handlers; agents (`.claude/agents/*.md`) use YAML frontmatter (name, description, model, tools, permissionMode, maxTurns, skills, mcpServers, hooks, memory, isolation, etc.); skills (`.claude/skills/<name>/SKILL.md`) declare name/description frontmatter + markdown instructions; MCP servers extend tools/data; CLAUDE.md is persistent instructions (not enforced); permission model is Deny > Ask > Allow (enforced by Claude Code, not by instructions).

**Evidence:**

### Installation & Authentication
- **Source:** Official Claude Code docs via search results, multiple 2026 installation guides, verified 2026-09-02. Official Anthropic console at [console.anthropic.com](https://console.anthropic.com/).
- **Installation method:** Native installer (auto-updates, recommended); Homebrew and npm are legacy routes. [Install Claude Code 2026 guide](https://www.nxcode.io/resources/news/install-claude-code-setup-guide-2026).
- **Authentication:** OAuth (post-login, token stored locally, no re-auth per session unless token expires) OR API key (generate at console.anthropic.com → Settings → API Keys; set via `claude login` interactively or env var `ANTHROPIC_API_KEY`).
- **Requirements:** Claude Pro ($20/month), Max 5x ($100/month), Max 20x ($200/month), Team ($20/seat/month), Enterprise, or Console account (per-token billing).

### Supported Surfaces
- **CLI (Terminal):** Command-line interface, primary surface for SDLC workflows.
- **IDE:** VS Code extension, JetBrains IDEs (IntelliJ, WebStorm, etc.).
- **Desktop:** Claude Code Desktop app (separate from web).
- **Web:** Claude.ai + Claude Code integration.
- **Source:** [Claude Code features overview](https://code.claude.com/) and agent SDK docs (verified 2026-09-02).

### Hooks: Events, Settings Schema, Handlers
- **Official source:** [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks), verified 2026-09-02.
- **Hook locations:**
  - User-wide: `~/.claude/settings.json` (applies to all projects, not shareable)
  - Project-specific: `.claude/settings.json` (shareable in repo) or `.claude/settings.local.json` (gitignored)
  - Plugin/skill/subagent frontmatter (session-scoped)

- **Top-level settings.json structure:**
  ```json
  {
    "hooks": {
      "EventName": [
        {
          "matcher": "pattern",
          "hooks": [
            {
              "type": "command|http|mcp_tool|prompt|agent",
              // handler-specific fields
            }
          ]
        }
      ],
      "disableAllHooks": false
    }
  }
  ```

- **Lifecycle events (comprehensive list from official docs):**
  - **Session scope:** SessionStart, Setup, SessionEnd
  - **Per-turn scope:** UserPromptSubmit, UserPromptExpansion, Stop, StopFailure
  - **Tool execution scope:** PreToolUse (can block), PermissionRequest, PermissionDenied, PostToolUse, PostToolUseFailure, PostToolBatch
  - **Other:** Notification, MessageDisplay, SubagentStart, SubagentStop, TaskCreated, TaskCompleted, TeammateIdle, InstructionsLoaded, ConfigChange, CwdChanged, DirectoryAdded, FileChanged, WorktreeCreate, WorktreeRemove, PreCompact, PostCompact, PreModelSwitch, PostModelSwitch, Elicitation, ElicitationResult

- **Hook handler types:**
  1. **command:** Shell script; fields: `type`, `command`, `args` (exec form) or shell form, `shell` (bash/powershell), `async`, `asyncRewake`, `timeout`, `statusMessage`
  2. **http:** HTTP endpoint; fields: `type`, `url`, `headers`, `allowedEnvVars`
  3. **mcp_tool:** MCP server tool call; fields: `type`, `server`, `tool`, `input`
  4. **prompt:** LLM prompt; fields: `type`, `prompt`, `model`
  5. **agent:** Dispatch to subagent; fields: `type`, `prompt`, `model`

- **Matcher patterns:** Literal strings (exact match), regex (contains special chars), list (comma/pipe separated), wildcard (`*`, `""`, omitted = match all).

- **Common hook fields:** `type`, `if` (permission rule), `timeout` (seconds), `statusMessage`, `once` (skills only).

- **This repo's use:** `.claude/settings.json` (lines 4–27 in CLAUDE.md example) implements PostToolUse(Edit|Write) → verify.sh, PreToolUse(Bash) → guard.sh, SessionStart → context.sh.

### Agents (.claude/agents/*.md)
- **Official source:** [code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents), verified 2026-09-02.
- **Frontmatter (YAML, complete schema):**
  ```yaml
  ---
  name: agent-name                    # Required: lowercase + hyphens
  description: When to delegate       # Required: brief, <15k tokens combined
  tools: Read, Grep, Glob, Bash       # Optional: allowlist; inherits all if omitted
  disallowedTools: Write, Edit        # Optional: denylist (applied before tools)
  model: sonnet|opus|haiku|fable|inherit|<full-id>  # Optional: which model
  permissionMode: default|acceptEdits|auto|dontAsk|bypassPermissions|plan  # Optional
  maxTurns: 10                        # Optional: max agentic turns
  skills: [skill-name, ...]           # Optional: preload skills
  mcpServers: [server-name, ...]      # Optional: MCP servers
  hooks:                              # Optional: lifecycle hooks (PreToolUse, PostToolUse, Stop)
    PreToolUse: [{ matcher: "...", hooks: [...] }]
  memory: user|project|local          # Optional: persistent scope
  background: false                   # Optional: keep in background
  effort: low|medium|high|xhigh|max   # Optional: override session effort
  isolation: worktree                 # Optional: isolated git worktree
  color: red|blue|green|yellow|purple|orange|pink|cyan  # Optional: display color
  initialPrompt: "First turn text"    # Optional: auto-submit if launched as main
  experimental:                       # Optional: experimental features
    cacheTtl: 5m|1h                   # Prompt caching TTL
  ---
  System prompt in Markdown here.
  ```

- **This repo's agents:**
  - `.claude/agents/researcher.md` (Haiku, grounding researcher)
  - `.claude/agents/chapter-writer.md` (Sonnet, content author)
  - `.claude/agents/chapter-reviewer.md` (Sonnet, QA/review)
  All use minimal frontmatter (name, description, model, tools); body is system prompt.

### Skills (.claude/skills/<name>/SKILL.md)
- **Official source:** [code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills), verified 2026-09-02.
- **Frontmatter (YAML):**
  ```yaml
  ---
  name: skill-name                    # Required: lowercase, match folder name
  description: What and when          # Required: Claude uses to decide load
  # Optional fields (not fully documented in search results, but inferred):
  # context: scope for skill
  # agent: advanced execution control
  ---
  Markdown instructions Claude follows when skill is invoked.
  ```

- **Location:** `.claude/skills/<skill-name>/SKILL.md` + supporting files in the folder.
- **Invocation:** Via `/skill-name` explicit command or auto-loaded when Claude deems relevant.
- **Difference from CLAUDE.md:** Skills load on-demand; CLAUDE.md is always active. Skills are for packaged procedures; CLAUDE.md for persistent facts.

### MCP (Model Context Protocol)
- **Official source:** [MCP Spec](https://spec.modelcontextprotocol.io/) and Anthropic engineering blog [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), verified 2026-09-02.
- **Definition:** MCP is a protocol for connecting agents to external tools and data sources. MCP servers define typed tools and data (like service dependencies with contracts).
- **In Claude Code:** Declared in agent frontmatter `mcpServers: [server-name]` or referenced in settings.json hooks. Enables agents to call external APIs/databases without embedding the logic in CLAUDE.md.
- **Relation to skills:** MCP servers provide *tools* (typed capabilities); skills provide *knowledge* (step-by-step instructions). Complementary: a skill can teach Claude to use an MCP-provided tool.
- **This repo:** References MCP in CLAUDE.md (agentic subject's MCP servers) but does not configure MCP in settings.json (Python/ML context, not agentic scaffold yet).

### CLAUDE.md (Rules / Persistent Instructions)
- **Role:** System prompt and charter for the agent / project. Defines intent, scope, golden rules, model routing, escalation criteria, repository shape, gates, etc.
- **Enforcement:** **NOT enforced by Claude Code.** CLAUDE.md is instructions to the model (Claude reads and follows it); it does NOT change permissions or auto-block actions. Permissions are enforced by rules in `.claude/settings.json` and the permission model.
- **Structure (this repo example):**
  ```
  # CLAUDE.md — ds_ml_ai_starter
  [Supervisor role, job, golden rules, model routing, escalation, repo shape, gates]
  ```
- **Source:** [Configure permissions — Claude Code Docs](https://code.claude.com/docs/en/permissions), verified 2026-09-02. Quote: "Instructions in your prompt or CLAUDE.md shape what Claude tries to do, but they don't change what Claude Code allows."

### Permission Model
- **Precedence (official):** Deny > Ask > Allow. Rules are evaluated deny (first match blocks), then ask, then allow. A deny rule blocks a tool call even if a broader allow rule matches.
- **Types of rules:** Allow (permit), Ask (prompt user), Deny (block).
- **Enforcement:** Enforced by Claude Code, not by the model. Rules are checked on every tool call before execution.
- **Scope:** Rules in `.claude/settings.json` apply to current working directory and additional directories (files become readable without prompts; edits follow permission mode).
- **Permission modes:** default (asks on risky tools), acceptEdits, auto (classifier reviews), dontAsk, bypassPermissions, plan (shows plan without running).
- **Source:** [Configure permissions](https://code.claude.com/docs/en/permissions), verified 2026-09-02.

**Caveats / limits:**

1. **Hook event coverage:** The list of 30+ hook events is comprehensive (SessionStart through ElicitationResult), but new events may be added in patch releases. Verify against official docs before pinning a specific event in a script.

2. **Schema validation:** The settings.json schema is published by Anthropic (reference `$schema: "https://json.schemastore.org/claude-code-settings.json"` in the repo), but community schemas may lag. Always validate against official docs.

3. **Agent/skill frontmatter stability:** The YAML schema for agents (10+ fields) and skills (name/description core) is stable, but optional fields (e.g., `experimental.cacheTtl`) may evolve. Reference the official docs URL in chapter prose.

4. **MCP server discovery:** Anthropic publishes official skills and MCP servers on GitHub (anthropics/skills) and a curated list at mcpservers.org. Third-party servers are community-maintained; no central registry of all servers.

5. **Permission model interaction with agents:** Agents inherit the permission mode of the session unless overridden in their frontmatter `permissionMode` field. Subagent permissions cascade; a subagent's deny rule does NOT override parent's allow rule (parent's rules apply).

6. **CLAUDE.md vs hooks:** A common confusion: CLAUDE.md is *instructions* (what Claude should try to do); hooks are *automation* (what runs at specific events); permissions are *boundaries* (what Claude is allowed to do). All three are independent.

7. **Env var substitution in hooks:** Env vars in hook command are NOT automatically substituted; use `${CLAUDE_PROJECT_DIR}` (placeholder provided by Claude Code) or `$VAR` in shell form (shell expands). Exec form (`args`) does not expand.

**Recommendation:**

For SPEC-SDLC-0 and SPEC-SDLC-1 chapter writers:

1. **Installation steps:** Show native installer (primary), mention Homebrew/npm legacy. Verify OAuth and API key auth both work; note that OAuth is simpler for interactive CLI.

2. **Surfaces (for SDLC-0):** Show CLI in terminal (primary), mention IDE/desktop/web as alternatives. Explain that all surfaces share the same `.claude/settings.json` (project-shareable).

3. **Hooks (for SDLC-1 Theory):** Explain hooks as "lifecycle automation, like git hooks" — PreToolUse can gate commands (like pre-commit blocks bad code), PostToolUse runs checks (like post-merge rebuild). Show this repo's verify.sh and guard.sh as examples.

4. **Agents & skills (for SDLC-1 Theory):** Define agent as "specialized subagent with a charter + tools + model". Skill as "packaged procedure" (like a runbook). MCP as "external tool dependency". Map to Java analogies: agent ≈ worker thread + dependency injection, skill ≈ utility library, MCP ≈ service API.

5. **CLAUDE.md & permissions (for SDLC-1 Theory + SDLC-2 Worked):** Stress that CLAUDE.md is instructions (not enforced), settings.json rules are enforced (Deny > Ask > Allow). Show this repo's CLAUDE.md as living reference for charter/golden-rules.

6. **Schemas for SDLC-2:** Cite official docs for agent/skill/hook/settings schemas. Validate example `.claude/` files in scaffold against [code.claude.com schemas](https://code.claude.com/docs/en/); JSON validate settings.json with `jq . .claude/settings.json`.

7. **Version/date:** Pin Claude Code docs URL + check date (e.g., "verified 2026-09-02 against code.claude.com/docs/en/sub-agents"). Schemas evolve; always link to live official source.

---

**Date verified:** 2026-09-02
