import com.example.sdlcdemo.LuhnValidator;

/**
 * Ad-hoc smoke driver — NOT part of the shipped Maven project (it lives at the project root, outside
 * src/, and no build step references it). It exists only so the LuhnValidator logic can be exercised
 * without a JUnit runner in an environment that has javac/java but no Maven. It asserts the exact
 * same 9 cases that {@code src/test/java/com/example/sdlcdemo/LuhnValidatorTest.java} encodes as
 * JUnit tests, and prints one PASS/FAIL line per case.
 *
 * Reproduce (JDK 21+):
 *   javac -d out -Xlint:all src/main/java/com/example/sdlcdemo/LuhnValidator.java SmokeCheck.java
 *   java -cp out SmokeCheck
 */
public final class SmokeCheck {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        // acceptsKnownValidNumbers — valid Luhn strings (spaces are stripped by the validator)
        checkValue("acceptsKnownValidNumbers(\"4111111111111111\")", "4111111111111111", true);
        checkValue("acceptsKnownValidNumbers(\"79927398713\")", "79927398713", true);
        checkValue("acceptsKnownValidNumbers(\"4012 8888 8888 1881\")", "4012 8888 8888 1881", true);

        // rejectsKnownInvalidNumbers — well-formed digit strings that fail the checksum
        checkValue("rejectsKnownInvalidNumbers(\"4111111111111112\")", "4111111111111112", false);
        checkValue("rejectsKnownInvalidNumbers(\"1234567890123456\")", "1234567890123456", false);
        checkValue("rejectsKnownInvalidNumbers(\"79927398710\")", "79927398710", false);

        // rejectsNonDigitCharacters — contains characters that are neither digits, spaces, nor hyphens
        checkLabelled("rejectsNonDigitCharacters", LuhnValidator.isValid("1234abcd"), false);

        // rejectsEmptyString
        checkLabelled("rejectsEmptyString", LuhnValidator.isValid(""), false);

        // rejectsNull — the validator throws IllegalArgumentException on null input
        boolean threw = false;
        try {
            LuhnValidator.isValid(null);
        } catch (IllegalArgumentException expected) {
            threw = true;
        }
        report("rejectsNull (IllegalArgumentException thrown)", threw, true);

        System.out.println();
        System.out.println(passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static void checkValue(String label, String input, boolean expected) {
        report(label, LuhnValidator.isValid(input), expected);
    }

    private static void checkLabelled(String label, boolean actual, boolean expected) {
        report(label, actual, expected);
    }

    private static void report(String label, boolean actual, boolean expected) {
        boolean ok = actual == expected;
        if (ok) {
            passed++;
        } else {
            failed++;
        }
        System.out.println((ok ? "PASS  " : "FAIL  ") + label + " -> " + actual);
    }
}
