import numpy as np

class Tensor:
  ops = {}
  opsgpu = {}
  def __init__(self, data, gpu=False) -> None:
    self.gpu = gpu
    self.grad = None
    self._ctx = None
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
      # fill in the first grad with one
      # this is "implicit gradient creation"
      assert self.data.shape == (1,)
      self.grad = Tensor(np.ones(self.data.shape, dtype=self.data.dtype), gpu=self.gpu)

    assert(self.grad is not None)

    grads = self._ctx.backward(self._ctx, self.grad.data)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t,g in zip(self._ctx.parents, grads):
      if g is None:
        continue
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" %
          (self._ctx, g.shape, t.data.shape))
        assert(False)
      t.grad = Tensor(g)
      t.backward(False)

from inspect import signature
import pyopencl as cl
cl_ctx = cl.create_some_context(answers=[0,2])  # change if you don't have mac
cl_queue = cl.CommandQueue(cl_ctx)

class Function:
 
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, *x, **kwargs):
    op = self
    ctx = op(*x)
    # use default params
    params = signature(op.forward).parameters
    for p in params.values():
      if p.default is not p.empty:
        setattr(ctx, p.name, p.default)
    # overwrite with passed params
    for k, v in kwargs.items():
      setattr(ctx, k, v)
    ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs))
    ret._ctx = ctx
    return ret
  
def register(name, fxn, gpu=False):
  if gpu:
    Tensor.opsgpu[name] = fxn
  else:
    Tensor.ops[name] = fxn
  def dispatch(self, *x, **kwargs):
    f = (Tensor.opsgpu if self.gpu else Tensor.ops)[name]
    f.cl_ctx, f.cl_queue = cl_ctx, cl_queue
    return f.apply(f, self, *x, **kwargs)
  setattr(Tensor, name, dispatch)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return y*grad_output, x*grad_output
register('mul', Mul)


def buffer_new(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx, x):
  return buffer_new(ctx, x.shape)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ret = buffer_like(ctx, x)
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void mul(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] * b_g[gid];
    }
    """).build()
    prg.mul(ctx.cl_queue, [ret.size//4], None, x, y, ret)
    ctx.save_for_backward(x, y, prg)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    x,y,prg = ctx.saved_tensors
    gx = buffer_like(ctx, x)
    gy = buffer_like(ctx, y)
    prg.mul(ctx.cl_queue, [gx.size//4], None, y, grad_output, gx)
    prg.mul(ctx.cl_queue, [gy.size//4], None, x, grad_output, gy)
    return gx, gy
register('mul', Mul, gpu=True)

a = Tensor(np.array([2]), True)
b = Tensor(np.array([3]), True)
c = a.mul(b)
d = c.mul(b)
d.backward()
print(b.grad.data)
print(b.data)
data = np.empty(b.grad.shape, dtype=np.float32)
cl.enqueue_copy(cl_queue, data, b.grad.data)
print(data)