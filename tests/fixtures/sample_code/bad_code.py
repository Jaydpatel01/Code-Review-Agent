def complex_function(a, b):
    # Missing docstring (docs rule)
    if a:
        if b:
            if a > 0:
                if b > 0:
                    if a < 10:
                        print("Deep nesting")  # Nesting depth = 5 (triggers nesting rule)
    
    # Cyclomatic complexity check (>10)
    if a == 1: pass
    elif a == 2: pass
    elif a == 3: pass
    elif a == 4: pass
    elif a == 5: pass
    elif a == 6: pass
    elif a == 7: pass
    elif a == 8: pass
    elif a == 9: pass
    elif a == 10: pass
    elif a == 11: pass
    elif a == 12: pass
    
    return a + b

def mutable_defaults_func(x=[]):
    # Mutable default (logic rule)
    x.append(1)
    return x

def long_function():
    # Long function (>50 lines)
    print("1")
    print("2")
    print("3")
    print("4")
    print("5")
    print("6")
    print("7")
    print("8")
    print("9")
    print("10")
    print("11")
    print("12")
    print("13")
    print("14")
    print("15")
    print("16")
    print("17")
    print("18")
    print("19")
    print("20")
    print("21")
    print("22")
    print("23")
    print("24")
    print("25")
    print("26")
    print("27")
    print("28")
    print("29")
    print("30")
    print("31")
    print("32")
    print("33")
    print("34")
    print("35")
    print("36")
    print("37")
    print("38")
    print("39")
    print("40")
    print("41")
    print("42")
    print("43")
    print("44")
    print("45")
    print("46")
    print("47")
    print("48")
    print("49")
    print("50")
    print("51")
    print("52")

def check_magic_numbers():
    x = 42  # Magic number 42
    y = 0   # 0 is allowed
    z = 1   # 1 is allowed
    CONSTANT = 100 # Assigned to uppercase, allowed
    return x + y + z + CONSTANT

class BadClass:
    # Missing docstring for class
    def __init__(self):
        pass
        
    def public_method(self):
        # Missing docstring for public method
        pass
