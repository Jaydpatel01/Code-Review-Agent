// bad_code.js — triggers every tree-sitter static check branch

// Missing JSDoc → docs finding (LOW)
function noDoc(a, b) {
    return a + b;
}

// Deep nesting (5 levels) → MEDIUM complexity finding
function deepNesting(x) {
    if (x > 0) {
        if (x > 1) {
            if (x > 2) {
                if (x > 3) {
                    if (x > 4) {
                        console.log("deep");
                    }
                }
            }
        }
    }
}

// High cyclomatic complexity (> 10) → MEDIUM complexity finding
function highComplexity(a, b, c, d) {
    if (a > 0) { }
    if (b > 0) { }
    if (c > 0) { }
    if (d > 0) { }
    if (a && b) { }
    if (b && c) { }
    if (c && d) { }
    if (a || d) { }
    while (a > 0) { a--; }
    for (let i = 0; i < 10; i++) { }
    try { } catch (e) { }
    return a ? b : c;
}

// Magic number → INFO style finding
function useMagicNumber() {
    let timeout = 42;
    return timeout * 3;
}

// Long function (> 50 lines) → MEDIUM complexity finding
function longFunction() {
    let x = 0;
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
    return x;
}
