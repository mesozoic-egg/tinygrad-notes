import numpy as np
import pyopencl as cl
# Boilerplate for openCL
cl_ctx = cl.create_some_context(answers=[0,2])  # change if you don't have mac
cl_queue = cl.CommandQueue(cl_ctx)
class Tensor:
  def __mul__(self, other):
    out = cl.Buffer(cl_ctx, cl.mem_flags.WRITE_ONLY, 4) # output is a single integer, 1 integer takes up 4 bytes

    # Write the actual openCL GPU code that would do the calculation
    prg = cl.Program(cl_ctx, """
        __kernel void mul(
            __global const float *a_g, __global const float *b_g, __global float *res_g)
        {
          int gid = get_global_id(0);
          res_g[gid] = a_g[gid] * b_g[gid];
        }
        """).build()

    # The .build() step will create a method of the same name on the prg instance, and we execute it
    prg.mul(cl_queue, [out.size//4], None, self.data, other.data, out)
    return Tensor(out, (self, other))
  
  def __add__(self, other):
    out = cl.Buffer(cl_ctx, cl.mem_flags.WRITE_ONLY, 4) # output is a single integer, 1 integer takes up 4 bytes

    # Write the actual openCL GPU code that would do the calculation
    prg = cl.Program(cl_ctx, """
        __kernel void add(
            __global const float *a_g, __global const float *b_g, __global float *res_g)
        {
          int gid = get_global_id(0);
          res_g[gid] = a_g[gid] + b_g[gid];
        }
        """).build()

    # The .build() step will create a method of the same name on the prg instance, and we execute it
    prg.add(cl_queue, [out.size//4], None, self.data, other.data, out)
    return Tensor(out, (self, other))
  
  def __init__(self, data, children=(), requires_grad=True) -> None:
    self.data = data
    self._prev = set(children)

    # Convert grad to openCL buffer
    grad = np.zeros(1)
    grad = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=grad)

    self.grad = Tensor(grad, requires_grad=False) if requires_grad else None

    self._backward = lambda: None

  def mul(self, other):
    out = self * other
    def _backward():
      self.grad += other * out.grad
      other.grad += self * out.grad
    out._backward = _backward
    return out
    
  def build_topo(self):
    topo = []
    visited = set()
    def _build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          _build_topo(child)
        topo.append(v)
    _build_topo(self)
    return topo

  def backward(self):

    # Convert to openCL buffer
    grad = np.ones((1,))
    grad = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=grad)
    self.grad = Tensor(grad)
    tree = self.build_topo()
    for tensor in reversed(tree):
      tensor._backward()
  
  def __repr__(self):
    return f"<Tensor with data: {self.data}, grad: {self.grad}>"
  

a = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([2]))
a = Tensor(a)
b = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([3]))
b = Tensor(b)
c = a.mul(b)
d = c.mul(b)
d.backward()
print(a)
# print(b)