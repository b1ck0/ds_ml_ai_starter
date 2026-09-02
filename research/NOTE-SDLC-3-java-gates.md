# NOTE-SDLC-3: Java Quality-Gate Tools & .claude File Schemas (2026)

**Answer:** Current standard Java quality gates are JUnit 5 (5.14.3) for testing (Maven Surefire 3.5.6 stable), Checkstyle 14.1.0 (Maven plugin 3.6.0) for style, Spotless 2.46.1 (Maven) / 8.10.1 (Gradle) for formatting, SpotBugs 4.10.3.0 (Maven) / 6.5.11 (Gradle) for bug detection; commands are `mvn -q test`, `mvn checkstyle:check`, `mvn spotless:check`, `mvn spotbugs:check` (Maven), and `gradle test`, `gradle spotlessCheck`, `gradle spotbugsMain` (Gradle); `.claude/settings.json` schema matches official code.claude.com docs (hooks block with event keys, matcher groups, handler types: command/http/mcp_tool/prompt/agent); agent frontmatter (name, description, model, tools, permissionMode, etc.) and skill frontmatter (name, description) match official schemas verified 2026-09-02.

**Evidence:**

### JUnit 5 Testing (5.14.3)

- **Version & stability:** [JUnit 5 Release Notes](https://docs.junit.org/5.13.4/release-notes/), verified 2026-09-02. 5.14.3 released July 2025, latest in 5.x line; 6.1.3 (Aug 2026) is newer but marks breaking change.
- **Maven Surefire:** Version 3.5.6 stable (latest in 3.5.x); 3.6.0-M1 is milestone/pre-release. [Maven Surefire Plugin releases](https://maven.apache.org/surefire/maven-surefire-plugin/).
- **Gradle test:** Built-in; add `testImplementation 'org.junit.jupiter:junit-jupiter:5.14.3'` and `test { useJUnitPlatform() }`.

**pom.xml (Maven) example:**
```xml
<dependency>
  <groupId>org.junit.jupiter</groupId>
  <artifactId>junit-jupiter</artifactId>
  <version>5.14.3</version>
  <scope>test</scope>
</dependency>

<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-surefire-plugin</artifactId>
  <version>3.5.6</version>
</plugin>
```

**gradle.kts (Gradle) example:**
```kotlin
testImplementation("org.junit.jupiter:junit-jupiter:5.14.3")

tasks.test {
  useJUnitPlatform()
}
```

**Run commands:**
- Maven: `mvn -q test` (quiet, suppress non-essential output) or `mvn clean test`
- Gradle: `gradle test` or `./gradlew test` (Gradle Wrapper preferred)

### Checkstyle (Style Compliance)

- **Tool version:** Checkstyle 14.1.0 (standalone tool). [Checkstyle releases](https://github.com/checkstyle/checkstyle/releases), verified 2026-09-02.
- **Maven plugin:** apache-maven-checkstyle-plugin 3.6.0 (bundled with Checkstyle 9.3 by default; can upgrade at runtime). [Maven Checkstyle Plugin](https://maven.apache.org/plugins/maven-checkstyle-plugin/).
- **Gradle:** No official Checkstyle Gradle plugin from Checkstyle project; use `com.github.spotbugs.snom:spotbugs-gradle-plugin` for SpotBugs, or integrate Checkstyle via `checkstyle` task.

**pom.xml (Maven) example:**
```xml
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-checkstyle-plugin</artifactId>
  <version>3.6.0</version>
  <configuration>
    <configLocation>checkstyle.xml</configLocation>
  </configuration>
</plugin>
```

**Run command:**
- Maven: `mvn checkstyle:check`

### Spotless (Code Formatting)

- **Maven plugin:** spotless-maven-plugin 2.46.1 (stable, JRE 17+). [Spotless Maven](https://mvnrepository.com/artifact/com.diffplug.spotless/spotless-maven-plugin/2.46.1), verified 2026-09-02.
- **Gradle plugin:** com.diffplug.spotless 8.10.1 (latest, Aug 2026). [Gradle Plugin Registry](https://plugins.gradle.org/plugin/com.diffplug.spotless).

**pom.xml (Maven) example:**
```xml
<plugin>
  <groupId>com.diffplug.maven</groupId>
  <artifactId>spotless-maven-plugin</artifactId>
  <version>2.46.1</version>
  <configuration>
    <java>
      <googleJavaFormat/>
    </java>
  </configuration>
</plugin>
```

**build.gradle.kts (Gradle) example:**
```kotlin
plugins {
  id("com.diffplug.spotless") version "8.10.1"
}

spotless {
  java {
    googleJavaFormat()
  }
}
```

**Run commands:**
- Maven: `mvn spotless:apply` (fix), `mvn spotless:check` (verify)
- Gradle: `gradle spotlessApply`, `gradle spotlessCheck`

### SpotBugs (Bug Detection)

- **Maven plugin:** spotbugs-maven-plugin 4.10.3.0 (latest, published 2026-07-13). [SpotBugs Maven Plugin](https://spotbugs.github.io/spotbugs-maven-plugin/).
- **Gradle plugin:** com.github.spotbugs (snom) 6.5.11 (latest, Aug 2026). [Gradle SpotBugs Plugin](https://plugins.gradle.org/plugin/com.github.spotbugs).

**pom.xml (Maven) example:**
```xml
<plugin>
  <groupId>com.github.spotbugs</groupId>
  <artifactId>spotbugs-maven-plugin</artifactId>
  <version>4.10.3.0</version>
</plugin>
```

**build.gradle.kts (Gradle) example:**
```kotlin
plugins {
  id("com.github.spotbugs") version "6.5.11"
}

spotbugs {
  ignoreFailures = false
}
```

**Run commands:**
- Maven: `mvn spotbugs:check`
- Gradle: `gradle spotbugsMain` (or `spotbugsTest`)

### Claude Code `.claude/settings.json` Schema

- **Official source:** [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks), verified 2026-09-02. Schema reference: `"$schema": "https://json.schemastore.org/claude-code-settings.json"`.
- **Top-level structure (matching THIS repo's `.claude/settings.json` lines 1–28):**
  ```json
  {
    "$schema": "https://json.schemastore.org/claude-code-settings.json",
    "hooks": {
      "EventName": [
        {
          "matcher": "pattern",
          "hooks": [
            {
              "type": "command|http|mcp_tool|prompt|agent",
              "command": "...",
              "args": [],
              "shell": "bash|powershell",
              "async": false,
              "timeout": 600,
              "statusMessage": "..."
            }
          ]
        }
      ]
    },
    "disableAllHooks": false
  }
  ```

- **Matcher patterns:** Literal strings, regex (if non-alphanumeric), list (comma/pipe separated), wildcard (`*`, `""`, omitted).
- **Hook event types:** SessionStart, PreToolUse, PostToolUse, Stop, UserPromptSubmit, and 25+ others.
- **Handler types:** `command` (shell script), `http` (webhook), `mcp_tool` (MCP server), `prompt` (LLM prompt), `agent` (subagent).

**Validation:** `jq . .claude/settings.json` (checks JSON well-formedness); schema validation via IDE (VS Code + JSON schema extension).

**This repo's use (verified):**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          { "type": "command", "command": ".claude/hooks/verify.sh" }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": ".claude/hooks/guard.sh" }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          { "type": "command", "command": ".claude/hooks/context.sh" }
        ]
      }
    ]
  }
}
```

### Agent Frontmatter Schema (.claude/agents/*.md)

- **Official source:** [code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents), verified 2026-09-02.
- **YAML frontmatter (complete schema):**
  ```yaml
  ---
  name: agent-name                    # Required
  description: Purpose and scope      # Required
  model: sonnet|opus|haiku|fable|inherit|<full-id>  # Optional
  tools: Read, Grep, Glob, Bash       # Optional
  disallowedTools: Write, Edit        # Optional
  permissionMode: default|acceptEdits|auto|dontAsk|bypassPermissions|plan  # Optional
  maxTurns: 10                        # Optional
  skills: [skill-name]                # Optional
  mcpServers: [server-name]           # Optional
  hooks: { PreToolUse: [...] }        # Optional
  memory: user|project|local          # Optional
  background: false                   # Optional
  effort: low|medium|high|xhigh|max   # Optional
  isolation: worktree                 # Optional
  color: red|blue|green|...           # Optional
  initialPrompt: "..."                # Optional
  experimental: { cacheTtl: 5m|1h }   # Optional
  ---
  System prompt in Markdown.
  ```

**This repo's agents (verified against schema):**
- `.claude/agents/researcher.md` — name: researcher, model: haiku, tools: Read, Grep, Glob, Bash, WebFetch, WebSearch, Write (matches schema)
- `.claude/agents/chapter-writer.md` — name: chapter-writer, model: sonnet (matches)
- `.claude/agents/chapter-reviewer.md` — name: chapter-reviewer, model: sonnet (matches)

### Skill Frontmatter Schema (.claude/skills/<name>/SKILL.md)

- **Official source:** [code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills), verified 2026-09-02.
- **YAML frontmatter (core schema):**
  ```yaml
  ---
  name: skill-name                    # Required: lowercase, match folder
  description: When to use            # Required
  # Optional (inferred from docs):
  # context: scope
  # agent: advanced execution
  ---
  Markdown instructions.
  ```

**This repo's use:**
- `.claude/skills/research-brief/SKILL.md` (frontmatter: name, description; body: detailed guide for researcher agent)

### Validation Checklist

1. **settings.json:** Well-formed JSON; schema reference present; hook events match official list (SessionStart, PreToolUse, PostToolUse, etc.).
2. **Agent frontmatter:** Required fields (name, description) present; model/tools/permissions optional; if omitted, inherit defaults.
3. **Skill frontmatter:** Required fields (name, description); name matches folder name (lowercase).
4. **Gate commands:** Verify `mvn -q test`, `mvn checkstyle:check`, `mvn spotless:check`, `mvn spotbugs:check` and Gradle equivalents are spelled correctly in scripts.

**Caveats / limits:**

1. **Gradle Checkstyle plugin:** Gradle has no official Checkstyle plugin; Checkstyle is usually run via Maven or as a custom task. If needed, use `com.github.jk1.dependency-license-report:com.github.jk1.dependency-license-report.gradle.plugin` or manually invoke Checkstyle JAR.

2. **Spotless Gradle version jump:** Spotless Gradle went from 2.x (Maven-style) to 8.x (breaking change). Version 8.10.1 is current; older projects may pin to 2.x compatibility. Clarify in chapter if advising readers to upgrade.

3. **SpotBugs Gradle plugin namespace:** The official Gradle plugin ID is `com.github.spotbugs`, not `spotbugs`; incorrect ID will fail silently (plugin not found) or pull wrong version.

4. **Maven Surefire 3.6.0 milestone:** As of 2026-09-02, version 3.6.0-M1 is pre-release; do NOT pin for production (use 3.5.6 stable). 3.6.0 final release date unknown.

5. **JUnit 6.x breaking change:** JUnit 6.x is a major version; migration from 5.x is not drop-in. For SDLC-2 scaffold, stick to 5.14.3 for stability.

6. **Schema evolution:** Hook event types, agent field names, and skill fields may evolve in patch releases. Always reference official docs URLs + date in chapter prose. Do not invent undocumented fields.

7. **Checkstyle configuration file:** pom.xml references `<configLocation>checkstyle.xml</configLocation>`, which must exist in the project root or classpath. This is a separate concern from the Maven plugin version.

**Recommendation:**

For SPEC-SDLC-2 chapter (Java project scaffold):

1. **Quality gates to include:** JUnit 5 (testing) + Checkstyle (style) + Spotless (formatting) + SpotBugs (bugs). Omit others (Jacoco, PMD, Sonar) as out of scope for SDLC intro.

2. **Maven vs Gradle:** Show Maven `.pom.xml` as primary (reader's comfort zone); Gradle `build.gradle.kts` as secondary. Both must build + test + gate cleanly.

3. **Gate commands in hooks/verify.sh:**
   ```bash
   mvn -q test
   mvn checkstyle:check
   mvn spotless:check
   mvn spotbugs:check
   ```
   For Gradle:
   ```bash
   gradle test spotlessCheck spotbugsMain
   ```

4. **Schema validation:** Show how to validate `.claude/settings.json` and `.claude/agents/*.md` frontmatter:
   - `jq . .claude/settings.json` (JSON check)
   - `grep -A5 "^---" .claude/agents/researcher.md | grep -E "^(name|description|model|tools)" | head -4` (frontmatter check)

5. **Cited versions in chapter:** Pin exact versions in pom.xml / build.gradle.kts snippets (e.g., JUnit 5.14.3, Maven Surefire 3.5.6). Cite this NOTE for version source + date.

6. **Reference transcripts:** Show real `mvn clean test` and `gradle test` output (duration, test count, pass/fail). If sandbox unavailable, note limitation and provide command as verified reference.

7. **CLAUDE.md for Java project:** Include gates section in project CLAUDE.md:
   ```markdown
   ## Gates
   Entry: approved spec + grounding landed. Exit: mvn clean test passes, checkstyle/spotless/spotbugs check pass.
   ```

---

**Date verified:** 2026-09-02

**Comparison to this repo's scaffold:** This project's `.claude/` (researcher, chapter-writer, chapter-reviewer agents; verify/guard/context hooks) is a **content-authoring scaffold**. SPEC-SDLC-2 must teach a **Java-project scaffold** (adapting the pattern to JVM build tools). The difference: here, hooks run snippet checks and link validation; in Java, hooks run `mvn test` and `checkstyle:check`. The file schemas (settings.json, agent frontmatter) are identical and must match official docs.

**Schema alignment verified:** This repo's `.claude/settings.json` (lines 4–27) and agent frontmatter files match official schema documented at code.claude.com/docs/en as of 2026-09-02. No proprietary or repo-specific extensions found.
