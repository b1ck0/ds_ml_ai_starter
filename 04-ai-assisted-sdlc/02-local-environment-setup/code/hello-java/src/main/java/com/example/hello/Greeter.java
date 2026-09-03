package com.example.hello;

/**
 * The one class this starter project exists to build and test.
 *
 * <p>Deliberately small: SPEC-SDLC-0 is not teaching Java (the reader already knows it), it is
 * proving the toolchain -- JDK + Maven + JUnit -- builds and tests something real, end to end,
 * before Claude Code ever touches the project.
 */
public class Greeter {

    /**
     * Builds a welcome message for {@code name}.
     *
     * @param name the person to greet; must not be {@code null} or blank
     * @return a greeting string
     * @throws IllegalArgumentException if {@code name} is {@code null} or blank
     */
    public String greet(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        return "Hello, " + name.trim() + "! Welcome to AI-assisted Java development.";
    }
}
