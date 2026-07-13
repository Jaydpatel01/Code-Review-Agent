// bad_code.java — triggers every tree-sitter static check branch for Java

public class BadClass {

    // Missing JavaDoc → docs finding (LOW)
    public void noDoc(int a, int b) {
        System.out.println(a + b);
    }

    // Deep nesting (5 levels) → MEDIUM complexity
    public void deepNesting(int x) {
        if (x > 0) {
            if (x > 1) {
                if (x > 2) {
                    if (x > 3) {
                        if (x > 4) {
                            System.out.println("deep");
                        }
                    }
                }
            }
        }
    }

    // High cyclomatic complexity (> 10) → MEDIUM complexity
    public int highComplexity(int a, int b, int c, int d) {
        if (a > 0) { }
        if (b > 0) { }
        if (c > 0) { }
        if (d > 0) { }
        if (a > 1 && b > 1) { }
        if (b > 1 && c > 1) { }
        if (c > 1 && d > 1) { }
        if (a > 1 || d > 1) { }
        while (a > 0) { a--; }
        for (int i = 0; i < 10; i++) { }
        try { } catch (Exception e) { }
        return a > 0 ? b : c;
    }

    // Magic number → INFO style finding
    public int useMagicNumber() {
        int timeout = 42;
        return timeout * 3;
    }

    // Long function (> 50 lines) → MEDIUM complexity
    public void longMethod() {
        int x = 0;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        x++;
        System.out.println(x);
    }
}
