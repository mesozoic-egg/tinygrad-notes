import numpy as np
import pyopencl as cl

cl_ctx = cl.create_some_context(answers=[0,2])  # change if you don't have mac
cl_queue = cl.CommandQueue(cl_ctx)

def print_buffer(buffer, shape, prefix=""):
  if isinstance(buffer, cl.Buffer):
    data = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(cl_queue, data, buffer)
    print(f"{prefix + ' ' if prefix else ''}{data=}")
  else:
    print(f"{prefix + ' ' if prefix else ''}{buffer}")

def _move_data(data, shape, device):
  if device == 'CPU':
    if isinstance(data, list):
      return np.array(data, dtype=np.float32)
    elif isinstance(data, cl.Buffer):
      ret = np.empty(shape, dtype=np.float32)
      cl.enqueue_copy(cl_queue, data, ret)
      return ret
    elif isinstance(data, np.ndarray):
      return data
  elif device == 'GPU':
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
      ret = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
      return ret
    elif isinstance(data, cl.Buffer):
      return data
    elif isinstance(data, np.ndarray):
      ret = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
      return ret
    
    
def binary_op_cpu(op, x, y, ret):
  if op == 'ADD':
    ret = x + y
  elif op == 'MUL':
    ret = x * y
  return ret

def binary_op_gpu(op, x, y, ret):
  if op == 'ADD':
    code = '+'
  elif op == 'MUL':
    code = '*'

  prg = cl.Program(cl_ctx, f"""
      __kernel void binary_op(
          __global const float *a_g, __global const float *b_g, __global float *res_g)
      {{
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] {code} b_g[gid];
      }}
      """).build()
  prg.binary_op(cl_queue, [ret.size//4], None, x, y, ret)
  return ret

class Tensor:
  def __init__(self, data, shape, device, requires_grad):
    self.device = device
    self.data = _move_data(data, shape, device)
    self.shape = shape
    self.requires_grad = requires_grad
    self._ctx = None
    self.grad = None

  def __repr__(self):
    return f"<Tensor {self.data!r} with grad {(self.grad.data if self.grad else None)!r}>"
  
  def __str__(self):
    return self.__repr__()
  
  def build_topo(self):
    topo = []
    visited = set()
    def _build_topo(v):
      if v not in visited:
        visited.add(v)
        if v._ctx:
          for parent in v._ctx.parents:
            _build_topo(parent)
        topo.append(v)
    _build_topo(self)
    return topo

  def backward(self):
    self.grad = Tensor(np.ones(self.shape, dtype=np.float32), self.shape, self.device, False)
    tree = self.build_topo()
    for tensor in reversed(tree):
      if tensor._ctx and any([t.requires_grad for t in tensor._ctx.parents]):
        grads = tensor._ctx.backward(tensor.grad)
        grads = [Tensor(g, self.shape, self.device, False) for g in grads]
        for t, g in zip(tensor._ctx.parents, grads):
          t.grad = g if t.grad is None else (t.grad + g)

class Function:
  def __init__(self, device, requires_grad, x, y):
    self.device = device
    self.requires_grad = requires_grad
    self.x = x
    self.y = y
    self.saved_tensors = []
    self.parents = [x, y]
    if device == 'CPU':
      self.buffer = lambda shape: np.ndarray(shape) 
    elif device == 'GPU':
      self.buffer = lambda shape: cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, 4*np.prod(shape))

  @classmethod
  def apply(cls, x, y):
    device = x.device
    requires_grad = x.requires_grad
    ctx = cls(device, requires_grad, x, y)
    ret_data = ctx.forward(x, y)
    ret = Tensor(ret_data, x.shape, device, requires_grad)
    ret._ctx = ctx
    return ret
  
  def save_for_backward(self, x, y):
    self.saved_tensors.extend([x, y])
  
  @property
  def binary_op(self):
    if self.device == 'CPU':
      return binary_op_cpu
    elif self.device == 'GPU':
      return binary_op_gpu
    
class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return ctx.binary_op('MUL', x.data, y.data, ctx.buffer(x.shape))

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = ctx.binary_op('MUL', grad_output.data, y.data, ctx.buffer(x.shape)) if x.requires_grad else None
    grad_y = ctx.binary_op('MUL', grad_output.data, x.data, ctx.buffer(x.shape)) if y.requires_grad else None
    return grad_x, grad_y

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return ctx.binary_op('ADD', x.data, y.data, ctx.buffer(x.shape))

  def backward(ctx, grad_output):
    return grad_output, grad_output

def register(name, fxn):
  def dispatch(x, y):
    return fxn.apply(x, y)
  setattr(Tensor, name, dispatch)

ops = {
  "__add__": Add,
  "__mul__": Mul,
}
for op, fxn in ops.items():
  register(op, fxn)

a = Tensor([2], (1,), "CPU", True)
b = Tensor([3], (1,), "CPU", True)
c = a * b
d = c * b
d.backward()
print_buffer(d.data, (1,))
print_buffer(a.grad.data, (1,))
print_buffer(b.grad.data, (1,))