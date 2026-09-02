package com.example.sdlcdemo;

/**
 * Validates a payment-card-style number against the Luhn checksum
 * [source: Luhn algorithm — Wikipedia](https://en.wikipedia.org/wiki/Luhn_algorithm) (checked
 * 2026-09-02).
 *
 * <p>This is the production code for {@code docs/features/FEATURE-1-luhn-validator.md}: reject a
 * card number before it ever reaches the payment gateway if it fails the checksum, instead of
 * paying a gateway round-trip (and a decline fee) to find out.
 *
 * <p>The checksum, read from the rightmost digit: double every second digit; if doubling pushes a
 * digit above 9, subtract 9 (equivalent to summing its own two digits); sum all digits; the number
 * is valid iff the total is divisible by 10.
 */
public final class LuhnValidator {

    private LuhnValidator() {
        // utility class — no instances
    }

    /**
     * Returns {@code true} iff {@code candidate} is a non-empty digit string (spaces and hyphens
     * are stripped first, so "4012 8888 8888 1881" and "4012-8888-8888-1881" are both accepted
     * forms) that satisfies the Luhn checksum.
     *
     * @throws IllegalArgumentException if {@code candidate} is {@code null}
     */
    public static boolean isValid(String candidate) {
        if (candidate == null) {
            throw new IllegalArgumentException("candidate must not be null");
        }
        String digitsOnly = candidate.replace(" ", "").replace("-", "");
        if (digitsOnly.isEmpty() || !digitsOnly.chars().allMatch(Character::isDigit)) {
            return false;
        }

        int sum = 0;
        boolean doubleThisDigit = false;
        for (int i = digitsOnly.length() - 1; i >= 0; i--) {
            int digit = digitsOnly.charAt(i) - '0';
            if (doubleThisDigit) {
                digit *= 2;
                if (digit > 9) {
                    digit -= 9;
                }
            }
            sum += digit;
            doubleThisDigit = !doubleThisDigit;
        }
        return sum % 10 == 0;
    }
}
