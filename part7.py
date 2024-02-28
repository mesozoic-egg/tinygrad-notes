"""
Let's implement an autograd library that can handle multiplication:

a = Tensor(np.array([2]))
b = Tensor(np.array([3]))
c = a.mul(b)
c.backward()

We will expect a.grad to be [3] and b.grad = to be [2]
"""
import numpy as np

class Tensor:
  def __init__(self, data, children=()) -> None:
    self.data = data
    self._prev = set(children)
    self.grad = np.zeros(self.data.shape)
    self._backward = lambda: None

  def __repr__(self):
    return f"<Tensor with data: {self.data}, grad: {self.grad}>"
  
  def mul(self, other):
    out = Tensor(self.data * other.data, (self, other))
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = np.ones(self.data.shape)
    for v in reversed(topo):
      v._backward()

a = Tensor(np.array([2,3]))
b = Tensor(np.array([3,4]))
c = a.mul(b)
c.backward()
print(a)
print(b)