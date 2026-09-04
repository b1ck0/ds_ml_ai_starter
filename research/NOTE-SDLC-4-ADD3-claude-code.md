# NOTE-SDLC-4-ADD3-claude-code

Grounding notes for the Claude Code walkthrough (SPEC-SDLC-4-ADDENDUM-3). All facts verified against official Claude Code documentation at https://code.claude.com/docs as of 2026-09-04.

## 1. Installing Claude Code

**Answer:** Three official install methods; native installer recommended (auto-updates). The `claude` command starts it after installation.

**Evidence:**

From https://code.claude.com/docs/en/overview (accessed 2026-09-04):

> To install Claude Code, use one of the following methods:
>
> **Native Install (Recommended)**
> macOS, Linux, WSL: `curl -fsSL https://claude.ai/install.sh | bash`
> Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`
> Windows CMD: `curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd`
> Native installations automatically update in the background to keep you on the latest version.

Alternative methods:
- **Homebrew:** `brew install --cask claude-code` (does not auto-update; requires `brew upgrade claude-code` manually)
- **WinGet:** `winget install Anthropic.ClaudeCode` (does not auto-update)
- **Linux package managers:** apt, dnf, or apk on Debian, Fedora, RHEL, and Alpine

From https://code.claude.com/docs/en/overview (accessed 2026-09-04):

> Then start Claude Code in any project. Replace `your-project` with the path to a project directory on your machine:
> ```
> cd your-project
> claude
> ```

**macOS specifics:** The quickstart (https://code.claude.com/docs/en/quickstart, accessed 2026-09-04) notes:
> Git for Windows is recommended on native Windows so Claude Code can use the Bash tool. If Git for Windows is not installed, Claude Code uses PowerShell as the shell tool instead. WSL setups do not need Git for Windows.

**Caveats:** Native vs. Homebrew auto-update behavior differs—native is automatic; Homebrew and WinGet require manual updates.

---

## 2. Signing In / Authentication

**Answer:** First-run flow prompts user to authenticate in a browser. Supports Claude Pro/Max/Team/Enterprise subscriptions, Claude Console (API), and third-party cloud providers (Amazon Bedrock, Google Cloud, Microsoft). API key via `ANTHROPIC_API_KEY` env var bypasses browser prompt.

**Evidence:**

From https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> Claude Code requires an account to use. Start an interactive session with the `claude` command and you'll be prompted to log in on first use:
> ```
> claude
> ```
> For Claude subscription or Console accounts, follow the prompts to complete authentication in your browser. If you've set the `ANTHROPIC_API_KEY` environment variable, Claude Code skips the login prompt and asks you to approve the key instead. To switch accounts later or re-authenticate, type `/login` inside the running session

Account options supported:
- Claude Pro, Max, Team, or Enterprise (recommended)
- Claude Console (API access with pre-paid credits; auto-creates "Claude Code" workspace)
- Amazon Bedrock, Google Cloud Agent Platform, Microsoft Foundry
- Self-hosted Claude apps gateway with corporate SSO

From https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> Once logged in, your credentials are stored and you won't need to log in again. Learn more in Credential Management.

**Caveats:** On first login with Console account, a "Claude Code" workspace is automatically created for cost tracking. Third-party cloud providers and self-hosted gateways require pre-configuration by admin.

---

## 3. Starting a Session in a Project (CLAUDE.md Reading)

**Answer:** User runs `cd` into project directory and executes `claude`. Claude Code reads `CLAUDE.md` files from the project root and parent directories at session start to load persistent instructions.

**Evidence:**

From https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> Open your terminal in any project directory and start Claude Code:
> ```
> cd /path/to/your/project
> claude
> ```
> Replace `/path/to/your/project` with the path to the project you want to work on.
> You'll see the Claude Code prompt with the version, current model, and working directory shown above it.

From https://code.claude.com/docs/en/memory (accessed 2026-09-04):

> CLAUDE.md files are markdown files that give Claude persistent instructions for a project, your personal workflow, or your entire organization. You write these files in plain text; Claude reads them at the start of every session.

CLAUDE.md load order (broadest to most specific):
1. Managed policy: `/Library/Application Support/ClaudeCode/CLAUDE.md` (macOS), `/etc/claude-code/CLAUDE.md` (Linux/WSL), `C:\Program Files\ClaudeCode\CLAUDE.md` (Windows)
2. User instructions: `~/.claude/CLAUDE.md`
3. Project instructions: `./CLAUDE.md` or `./.claude/CLAUDE.md`
4. Local (personal-only): `./CLAUDE.local.md`

From https://code.claude.com/docs/en/memory (accessed 2026-09-04):

> Claude Code loads `CLAUDE.md` and `CLAUDE.local.md` from your current working directory and every directory above it. All discovered files are concatenated into context rather than overriding each other.

**Caveats:** Files in subdirectories load on demand when Claude reads files in those directories, not at session start.

---

## 4. Sub-Agents

**Answer:** `.claude/agents/*.md` files define sub-agents with YAML frontmatter (`name`, `description`, `tools`, `model`, etc.). Invoked three ways: natural language (Claude decides), @-mention (force specific agent), or CLI flag (`claude --agent agent-name`). Sub-agents run in isolated context windows and can be configured with persistent memory.

**Evidence:**

From https://code.claude.com/docs/en/sub-agents (accessed 2026-09-04):

> Ask Claude to create a subagent by describing what you want:
> ```
> Create a personal code-improver subagent in ~/.claude/agents/ that scans
> files and suggests improvements for readability, performance, and best
> practices. Make it read-only and have it use Sonnet.
> ```
> Claude writes a markdown file with YAML frontmatter:
> ```markdown
> ---
> name: code-improver
> description: Scans files and suggests improvements for readability, performance, and best practices
> tools: Read, Grep, Glob
> model: sonnet
> ---
> ```

Three invocation methods from https://code.claude.com/docs/en/sub-agents (accessed 2026-09-04):

1. **Natural Language:** `Use the code-reviewer subagent to look at my recent changes` (Claude decides if relevant)
2. **@-Mention:** `@"code-reviewer (agent)" review the auth changes` or `@agent-code-reviewer`
3. **Session-wide (CLI or settings):** `claude --agent code-improver` or in `.claude/settings.json`:
   ```json
   {
     "agent": "code-improver"
   }
   ```

Storage locations (priority order):
1. Managed settings
2. `--agents` CLI flag (current session)
3. `.claude/agents/` (project-specific; check into version control)
4. `~/.claude/agents/` (personal; all projects)
5. Plugin `agents/` directory

Key configuration fields (from frontmatter):
- `name` (required): unique identifier
- `description` (required): when Claude should delegate
- `tools`: allowed tools (Read, Write, Edit, Bash, etc.)
- `model`: which model to run (sonnet, opus, haiku, or inherit)
- `permissionMode`: permission behavior
- `memory`: persistent memory scope (user, project, or local)
- `isolation`: run in isolated worktree

**Caveats:** Sub-agents run in isolated context; auto memory from main conversation is not loaded into sub-agents (except via fork).

---

## 5. Hooks

**Answer:** `.claude/hooks/` scripts + `.claude/settings.json` wire `PreToolUse` and `PostToolUse` events that execute shell commands, HTTP calls, or MCP tool calls at specific lifecycle points (before/after tool use). Hooks receive JSON stdin and exit codes determine permission decisions.

**Evidence:**

From https://code.claude.com/docs/en/hooks (accessed 2026-09-04):

> Hooks are user-defined shell commands, HTTP endpoints, MCP tool calls, LLM prompts, or subagents that execute automatically at specific points in Claude Code's lifecycle.

Hook lifecycle from https://code.claude.com/docs/en/hooks (accessed 2026-09-04):

> Hooks fire at three cadences:
> - Once per session: `SessionStart`, `SessionEnd`
> - Once per turn: `UserPromptSubmit`, `Stop`, `StopFailure`
> - On every tool call: `PreToolUse`, `PostToolUse` (except `EndConversation` calls)

Hook handler types:
1. **Command hooks** (`type: "command"`): shell commands receiving JSON on stdin
2. **HTTP hooks** (`type: "http"`): POST requests to a URL
3. **MCP tool hooks** (`type: "mcp_tool"`): calls to MCP server tools
4. **Prompt hooks** (`type: "prompt"`): single-turn LLM evaluation
5. **Agent hooks** (`type: "agent"`): subagent-based verification

Configuration structure from https://code.claude.com/docs/en/hooks (accessed 2026-09-04):

> Hooks are defined in JSON with three levels of nesting:
> 1. Hook event (e.g., `PreToolUse`, `Stop`)
> 2. Matcher group (e.g., filter to specific tools like "Bash")
> 3. Hook handlers (the actual command, HTTP endpoint, etc.)

Example (blocking `rm -rf`) from docs:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(rm *)",
            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-rm.sh",
            "args": []
          }
        ]
      }
    ]
  }
}
```

Hook locations and scope from https://code.claude.com/docs/en/hooks (accessed 2026-09-04):

| Location | Scope | Shareable |
|----------|-------|-----------|
| `~/.claude/settings.json` | All projects | No |
| `.claude/settings.json` | Single project | Yes (commit to repo) |
| `.claude/settings.local.json` | Single project | No (gitignored) |
| Plugin `hooks/hooks.json` | When plugin enabled | Yes |
| Skill/Subagent frontmatter | That component's session | Yes |

Exit codes from https://code.claude.com/docs/en/hooks (accessed 2026-09-04):

> - Exit 0: Success; Claude Code reads JSON output for decisions
> - Exit 2: Blocking error; prevents action (e.g., tool call blocked)
> - Other codes: Non-blocking; action proceeds

**Caveats:** Hooks operate outside CLAUDE.md instructions—they enforce hard rules regardless of Claude's decisions. Commands receive structured JSON stdin; scripts must parse and respond with JSON for permission decisions.

---

## 6. Permission Modes

**Answer:** Claude Code has tiered permission system. **Manual mode** asks before each action; **Auto mode** uses a classifier to review actions without prompting. Users can approve specific actions with "Yes, and don't ask again" to save permanent allow rules. Permission modes configurable via settings, env var, or CLI flag.

**Evidence:**

From https://code.claude.com/docs/en/permissions (accessed 2026-09-04):

> Claude Code supports fine-grained permissions so that you can specify exactly what the agent is allowed to do and what it can't. You can check permission settings into version control to share them with every developer in your organization, and each developer can customize their own.

Permission system (from same source):

> Claude Code uses a tiered permission system to balance power and safety. The table shows, for each tool type, whether Manual mode asks before the action runs. The other permission modes change which of these ask you; in auto mode a classifier reviews actions instead of you...

From https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> Auto mode is the built-in starting permission mode for interactive terminal sessions on Pro, Max, and Team plans: a classifier reviews actions instead of you, and Claude edits most files and runs most commands without asking you. On other plans, Manual mode is the built-in starting permission mode.

User approval persistence from https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> If it asks before making the change, select **Yes** to approve. 

And from settings page (accessed 2026-09-04):

> When you ask permission to run a Bash command and you choose "Yes, and don't ask again", Claude Code saves that approval as an `allow` rule in `.claude/settings.local.json`.

Permission configuration from https://code.claude.com/docs/en/settings (accessed 2026-09-04):

> Claude Code reads settings from four files, and an organization can also deliver managed settings from the claude.ai console. Each source has a scope...

Modes can be set via:
- `/config` menu (in session)
- `.claude/settings.json` or `~/.claude/settings.json`
- `claude --permission-mode` CLI flag
- Managed settings (organization enforces)

**Caveats:** Permission modes vary by plan (Pro/Max/Team default to Auto; others default to Manual). Permission approvals from local settings file apply without workspace trust (unlike committed `.claude/settings.json`). Managed settings override all user settings for certain keys.

---

## 7. First-Session Flow (Newcomer-Focused)

**Answer:** (1) Install via native installer. (2) Run `claude` to trigger browser login. (3) `cd project && claude` to start session. (4) Describe task in natural language. (5) Review diff/plan shown by Claude. (6) Approve actions when prompted (or Auto mode auto-approves). (7) Claude commits and returns to prompt.

**Evidence:**

Installation from https://code.claude.com/docs/en/overview (accessed 2026-09-04):

> To install Claude Code, use one of the following methods:
> **Native Install (Recommended)**
> macOS, Linux, WSL: `curl -fsSL https://claude.ai/install.sh | bash`
> Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`

Login from https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> Claude Code requires an account to use. Start an interactive session with the `claude` command and you'll be prompted to log in on first use:
> ```
> claude
> ```
> For Claude subscription or Console accounts, follow the prompts to complete authentication in your browser.

Starting a session from same source:

> Open your terminal in any project directory and start Claude Code:
> ```
> cd /path/to/your/project
> claude
> ```

Task flow from https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> **Step 4: Ask your first question**
> Let's start with understanding your codebase. Try one of these commands:
> ```
> what does this project do?
> ```
> Claude will analyze your files and provide a summary.

> **Step 5: Make your first code change**
> Now let's make Claude Code do some actual coding. Try a simple task:
> ```
> add a hello world function to the main file
> ```
> Claude Code finds the appropriate file and shows you the change. If it asks before making the change, select **Yes** to approve.

Git integration from https://code.claude.com/docs/en/quickstart (accessed 2026-09-04):

> **Step 6: Use Git with Claude Code**
> Claude Code makes Git operations conversational:
> ```
> commit my changes with a descriptive message
> ```

**Caveats:** Permission mode (Manual vs. Auto) depends on subscription plan; Manual requires approval per action, Auto uses classifier. CLAUDE.md is optional but recommended for persistent team instructions.

---

## Recommendations for the Walkthrough

1. **Install section:** Emphasize native installer for auto-updates. Show shell/PowerShell distinction for Windows. Mention Git for Windows recommendation.
2. **Auth section:** Show browser-based flow as primary; note ANTHROPIC_API_KEY as env-var alternative.
3. **Project setup:** Mention CLAUDE.md discovery; show simple example (e.g., build commands, coding standards).
4. **Subagents:** Don't over-complicate; show one simple example (e.g., a code-reviewer agent). Mention `.claude/agents/` directory but keep it brief.
5. **Hooks & permissions:** Keep separate; hooks are for hard enforcement (security), permissions are for user/auto mode configuration.
6. **First task:** Start with read-only (e.g., "what does this project do?") before any writes. Then a simple add (e.g., "add a TODO comment").
7. **Avoid:** Never claim a version, command, or URL from memory. Always cite the official docs and date checked.

---

**Dates checked:** All sources verified 2026-09-04.
**Official source:** https://code.claude.com/docs (primary); https://platform.claude.com/docs (secondary for Managed Agents API cross-reference).
