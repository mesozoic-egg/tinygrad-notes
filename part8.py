"""
We will use the same example, but this time, we would want to be able to run
the code on the GPU

a = Tensor(np.array([2]))
b = Tensor(np.array([3]))
c = a.mul(b)
d = c.mul(b)
d.backward()
"""

"""
First change we will make is to isolate the definition of Mul
This arrangement allows us to have a separate implementation for GPU. Below
is the interface design
"""
import numpy as np

class Tensor:
  def __init__(self, data) -> None:
    self.data = data
    self.grad = None

  def backward(self):
    pass

def register(name, fxn):
  setattr(Tensor, name, fxn)

class Mul:
  def forward(self, x, y):
    return x * y
  
  def backward(self, output):
    return
  
class MulGPU:
  def forward(self, x, y):
    """
    You can embed openCL or CUDA code here
    __kernel void mul(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] * b_g[gid];
    }
    """
    return
  
  def backward(self, output):
    return
  
"""
We now look at how we can actually fill in the gap between a Mul op and the tensor
"""

class Tensor:
  ops = {}
  def __init__(self, data) -> None:
    self.data = data
    self.grad = None
    self._ctx = None

  def backward(self, allow_fill=True):
    if self._ctx is None:
      return
    if self.grad is None and allow_fill:
      self.grad = Tensor(np.ones(self.data.shape))
    grads = self._ctx.backward(self.grad) # backward on op
    print(f"{grads=}")
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t, g in zip(self._ctx.parents, grads):
      if t.grad is not None:
        t.grad = Tensor(t.grad.data + g)
      else:
        t.grad = Tensor(g)
      t.backward(False) # backward on tensor

  def __repr__(self):
    return f"<Tensor with data: {self.data}, grad: {self.grad}>"


def register(name, fxn):
  Tensor.ops[name] = fxn
  def dispatch(self, y):
    f = Tensor.ops[name]
    ret = f.apply(self, y)
    return ret
  setattr(Tensor, name, dispatch)

class Function:
  def __init__(self, *tensors):
    self.parents = tensors
  
  @classmethod
  def apply(cls, x, y):
    op = cls(x, y)
    data = op.forward(x.data, y.data)
    ret = Tensor(data)
    ret._ctx = op
    return ret

class Mul(Function):
  def forward(self, x, y):
    return x * y

  def backward(self, output):
    x, y = self.parents
    return y.data * output.data, x.data * output.data

register('mul', Mul)
a = Tensor(np.array([2]))
b = Tensor(np.array([3]))
c = a.mul(b)
d = c.mul(b)
d.backward()
print(a) # Expect a to have gradient of 9
print(b) # Expect b to have gradient of 12