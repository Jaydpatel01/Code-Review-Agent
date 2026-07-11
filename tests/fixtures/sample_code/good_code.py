"""A module containing good code that passes all static analysis checks."""

def simple_function(a, b):
    """Adds two numbers together."""
    if a > 0:
        return a + b
    return b

def no_mutable_defaults(x=None):
    """Appends 1 to x safely."""
    if x is None:
        x = []
    x.append(1)
    return x

class GoodClass:
    """A well-documented class."""
    
    def __init__(self):
        """Initializes the GoodClass."""
        self.value = 1
        
    def public_method(self):
        """A well-documented public method."""
        return self.value
