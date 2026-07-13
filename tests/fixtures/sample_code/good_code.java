// good_code.java — clean Java that should produce zero complexity/nesting findings

public class GoodClass {

    /**
     * Adds two integers.
     * @param a first number
     * @param b second number
     * @return sum
     */
    public int add(int a, int b) {
        return a + b;
    }

    /**
     * Returns the larger of two integers.
     * @param x first value
     * @param y second value
     * @return the maximum value
     */
    public int max(int x, int y) {
        if (x > y) {
            return x;
        }
        return y;
    }
}
