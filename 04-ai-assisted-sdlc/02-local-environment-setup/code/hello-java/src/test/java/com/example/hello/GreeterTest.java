package com.example.hello;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GreeterTest {

    private final Greeter greeter = new Greeter();

    @Test
    @DisplayName("greets a normal name")
    void greetsByName() {
        assertEquals(
                "Hello, Ada! Welcome to AI-assisted Java development.",
                greeter.greet("Ada"));
    }

    @Test
    @DisplayName("trims surrounding whitespace before greeting")
    void trimsWhitespace() {
        assertEquals(
                "Hello, Grace! Welcome to AI-assisted Java development.",
                greeter.greet("  Grace  "));
    }

    @Test
    @DisplayName("rejects a blank name")
    void rejectsBlankName() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> greeter.greet("   "));
        assertEquals("name must not be blank", ex.getMessage());
    }

    @Test
    @DisplayName("rejects a null name")
    void rejectsNullName() {
        assertThrows(IllegalArgumentException.class, () -> greeter.greet(null));
    }
}
