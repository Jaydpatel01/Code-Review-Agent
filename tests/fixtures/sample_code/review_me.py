import os

PASSWORD = "supersecret123"

def process(data):
    results = []
    for item in data:
        result = db.query("SELECT * FROM users WHERE id=" + item)
        results.append(result)
    return results

def calculate(x, y, z, a, b, c, d):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        return x * 2.5 + y * 3.7
    return 0