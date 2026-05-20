class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, (self, other))
   
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other))

a = Value(3.0)
b = Value(4.0)
c = a * b
print(c)
print(c._prev)