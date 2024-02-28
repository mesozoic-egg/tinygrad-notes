"""
a = Tensor(np.array([2]))
b = Tensor(np.array([3]))
c = a.mul(b)
d = c.mul(b)
d.backward()
"""

"""
Expanding on previous, we can implement the GPU support with openCL
"""
import numpy as np
import pyopencl as cl

cl_ctx = cl.create_some_context(answers=[0,2])  # change if you don't have mac
cl_queue = cl.CommandQueue(cl_ctx)

class Tensor:
  ops = {}
  opsgpu = {}
  def __init__(self, data, gpu=False) -> None:
    self.grad = None
    self._ctx = None
    self.gpu = gpu

    if isinstance(data, cl._cl.Buffer):
      self.gpu = True
      self.data = data
    elif self.gpu:
      self.data = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
      self.data.shape = data.shape
    else:
      self.data = data
  
  @property
  def shape(self):
    return self.data.shape

  def backward(self, allow_fill=True):
    if self._ctx is None:
      return
    if self.grad is None and allow_fill:
      self.grad = Tensor(np.ones(self.data.shape), self.gpu)
    grads = self._ctx.backward(self.grad.data) # backward on op
    print(f"{grads=}")
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t, g in zip(self._ctx.parents, grads):
      if t.grad is not None:
        if t.gpu:
          t.grad = Tensor(t.grad.add(g), True)
        else:
          t.grad = Tensor(t.grad.data + g)
      else:
        t.grad = Tensor(g)
      t.backward(False) # backward on tensor

  def __repr__(self):
    return f"<Tensor with data: {self.data}, grad: {self.grad}>"


def register(name, fxn, gpu=False):
  if gpu:
    Tensor.opsgpu[name] = fxn
  else:
    Tensor.ops[name] = fxn
  def dispatch(self, y):
    F = (Tensor.opsgpu if self.gpu else Tensor.ops)[name]
    F.cl_ctx, F.cl_queue = cl_ctx, cl_queue
    ret = F.apply(self, y)
    return ret
  setattr(Tensor, name, dispatch)

class Function:
  cl_ctx = None
  cl_queue = None
  def __init__(self, *tensors):
    self.parents = tensors
    self.prg = None
    self.saved_tensors = []
  
  def save_for_backward(self, *t):
    self.saved_tensors.extend(t)
  
  @classmethod
  def apply(cls, x, y):
    op = cls(x, y)
    op.save_for_backward(x.data, y.data)
    data = op.forward(x.data, y.data)
    ret = Tensor(data)
    ret._ctx = op
    return ret

class Mul(Function):
  def forward(self, x, y):
    return x * y

  def backward(self, output):
    x, y = self.saved_tensors
    return y * output, x * output

register('mul', Mul)

def buffer_new(op, shape):
  res_g = cl.Buffer(op.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(op, x):
  return buffer_new(op, x.shape)

class MulGPU(Function):
  def forward(self, x, y):
    ret = buffer_like(self, x)
    prg = cl.Program(self.cl_ctx, """
    __kernel void mul(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] * b_g[gid];
    }
    """).build()
    prg.mul(self.cl_queue, [ret.size//4], None, x, y, ret)
    self.prg = prg
    return ret

  def backward(self, output):
    x, y = self.saved_tensors
    prg = self.prg
    gx = buffer_like(self, x)
    gy = buffer_like(self, y)
    prg.mul(self.cl_queue, [gx.size //4], None, y, output, gx)
    prg.mul(self.cl_queue, [gy.size //4], None, x, output, gy)
    return gx, gy

  
register('mul', MulGPU, True)

a = Tensor(np.array([2]), True)
b = Tensor(np.array([3]), True)
c = a.mul(b)
d = c.mul(b)
d.backward()
print(a) # Expect a to have gradient of 9
print(b) # Expect b to have gradient of 12