package com.example.sdlcdemo;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * JUnit 5 (5.14.3, {@code org.junit.jupiter:junit-jupiter} — pinned in {@code pom.xml}, grounded
 * in research/NOTE-SDLC-3-java-gates.md) tests for {@link LuhnValidator}.
 *
 * <p>This file existed, failing, BEFORE {@link LuhnValidator} did — see
 * {@code docs/features/FEATURE-1-luhn-validator.md} and the reference transcript in
 * {@code ../../../artefacts/feature-loop-transcript.md} for the commit-order evidence.
 */
class LuhnValidatorTest {

    @ParameterizedTest
    @ValueSource(strings = {"4111111111111111", "79927398713", "4012 8888 8888 1881"})
    void acceptsKnownValidNumbers(String candidate) {
        assertTrue(LuhnValidator.isValid(candidate));
    }

    @ParameterizedTest
    @ValueSource(strings = {"4111111111111112", "1234567890123456", "79927398710"})
    void rejectsKnownInvalidNumbers(String candidate) {
        assertFalse(LuhnValidator.isValid(candidate));
    }

    @Test
    void rejectsNonDigitCharacters() {
        assertFalse(LuhnValidator.isValid("4111-11a1-1111-1111"));
    }

    @Test
    void rejectsEmptyString() {
        assertFalse(LuhnValidator.isValid(""));
    }

    @Test
    void rejectsNull() {
        assertThrows(IllegalArgumentException.class, () -> LuhnValidator.isValid(null));
    }
}
