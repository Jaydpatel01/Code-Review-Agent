// good_code.js — clean JS that should produce zero complexity/nesting findings

/**
 * Adds two numbers.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
function add(a, b) {
    return a + b;
}

/**
 * Returns the larger of two numbers.
 * @param {number} x
 * @param {number} y
 * @returns {number}
 */
function max(x, y) {
    if (x > y) {
        return x;
    }
    return y;
}
