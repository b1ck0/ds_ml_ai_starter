# NOTE-SDLC-1: Java Toolchain — JDK, Maven, Gradle, JUnit 5 (2026)

**Answer:** JDK 25 LTS (released Sept 2025) is the current recommended version; Maven 3.9.16 (stable, JDK 8+); Gradle 9.7.1 (stable, Aug 2026); JUnit 5 latest is 5.14.3 (Platform/Jupiter/Vintage unified); standard commands are `mvn archetype:generate` and `gradle init`, building with `mvn -q test` and `gradle test`.

**Evidence:**

### JDK 25 LTS
- **Source:** [Oracle Java SE Support Roadmap](https://www.oracle.com/java/technologies/java-se-support-roadmap.html), verified 2026-09-02
- **Quoted:** JDK 25 released Sept 2025, supported LTS versions as of April 2026 include JDK 25 (NFTC until Sept 2028), 21, 17, 11, 8; users on JDK 21 can upgrade to 25 during one-year overlap (until Sept 2026).
- **Installation:** Download from [Oracle Java SE Downloads](https://www.oracle.com/java/technologies/downloads/) — native installers for macOS, Windows, Linux; or via package managers (Homebrew, apt, yum).

### Maven 3.9.16
- **Source:** [Apache Maven Download](https://maven.apache.org/download.cgi), verified 2026-09-02
- **Quoted:** "Apache Maven 3.9.16 is the latest release and is recommended for all users. Maven 3.9+ requires JDK 8 or above to execute."
- **Archetype command:** `mvn archetype:generate -DgroupId=com.example -DartifactId=my-app` (standard form from Maven Archetypes documentation).

### Gradle 9.7.1
- **Source:** [Gradle 9.7.1 Release Notes](https://docs.gradle.org/current/release-notes.html), verified 2026-09-02; [Gradle Releases](https://gradle.org/releases/) confirms 9.7.1 (Aug 2026) as latest stable.
- **Init command:** `gradle init` with optional `--type java-application|java-library`, `--dsl kotlin|groovy`, `--test-framework junit|testng|spock` (from [Build Init Plugin docs](https://docs.gradle.org/current/userguide/build_init_plugin.html)).

### JUnit 5 (5.14.3)
- **Source:** [JUnit 5 Release Notes](https://docs.junit.org/5.13.4/release-notes/), verified 2026-09-02; GitHub [junit-team/junit-framework](https://github.com/junit-team/junit5/releases) shows 5.14.3 latest in 5.x line (released July 2025).
- **Quoted:** "JUnit 5.13.4 (July 21, 2025)… All experimental APIs promoted to maintained"; 5.14.3 includes unified Platform/Jupiter/Vintage.
- **Maven config:** Maven Surefire 3.5.x (stable) with `<groupId>org.junit.jupiter</groupId><artifactId>junit-jupiter</artifactId><version>5.14.3</version>`.
- **Gradle config:** JUnit 5 platform via `testImplementation 'org.junit.jupiter:junit-jupiter:5.14.3'` and `test { useJUnitPlatform() }`.

### Build/Test Commands
- **Maven:** `mvn -q test` (quiet flag suppresses non-essential output); full compile-test cycle is `mvn clean test`.
- **Gradle:** `gradle test` (default task); uses Gradle Wrapper `./gradlew test` (cross-platform preferred).
- **Source:** [Maven Surefire Plugin Usage](https://maven.apache.org/surefire/maven-surefire-plugin/usage.html), [Gradle Testing Documentation](https://docs.gradle.org/current/userguide/testing_overview.html).

**Caveats / limits:**

1. **JDK LTS overlap:** JDK 21 support ends Sept 2026 (one year after JDK 25 release); migrating from 21 to 25 is straightforward but pin the version in JAVA_HOME or toolchain config if the project must support 21.
2. **Maven vs Gradle parity:** Both are equally valid; Maven archetypes are slightly older/more conservative; Gradle's init is newer and more opinionated. Recommendation: match the reader's prior Java/build experience.
3. **JUnit 5 vs 6:** JUnit 6.1.3 released Aug 2026, but marks a breaking change (JUnit 5 → 6 migration); SPEC-SDLC pins to 5.14.3 for stability. Surefire 3.6.0-M1 is milestone; use 3.5.6 stable.
4. **Gradle Wrapper:** Best practice is to commit `.gradle/wrapper/gradle-wrapper.jar` and `.gradle/wrapper/gradle-wrapper.properties` so builds are reproducible without installing Gradle globally.

**Recommendation:**

For SPEC-SDLC-0 chapter writer:

1. **Pin JDK 25 in examples.** If testing in-sandbox, use JAVA_HOME or `.java-version` / `gradle.properties java.version = 21` for backward compat; capture real test output or note sandbox limitation.
2. **Use Maven 3.9.16 as primary, Gradle 9.7.1 as alternate.** Show both archetype and init commands; readers with Java background prefer Maven, but modern projects trend Gradle.
3. **Pin JUnit 5.14.3 in pom.xml and build.gradle.kts; use Maven Surefire 3.5.6 (stable, not M1).**
4. **Cite official Maven/Gradle/JUnit docs for dependency config** (not from memory). Verify snippets compile and test pass.
5. **Captured output:** If sandbox has JDK, show real `mvn clean test` and `gradle test` output (including test count, duration). If not, provide the commands as verified reference and note limitation.
6. **Gradle Wrapper:** Encourage committing the wrapper to git; mention `gradle wrapper --gradle-version 9.7.1`.

---

**Date verified:** 2026-09-02
