# Shapetracker, Symbolic and how they help upcasting (loop unrolling)

## expr_idxs

In tinygrad, there's a way to represent a tensor whose shape might change.
For example, a 2D grid tensor that can range from one row, to 100 rows, can
be represented as such:

```python
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

k = Variable("k", 1, 100)
st = ShapeTracker.from_shape((k, 3))
```

Intuitively, if you think the grid as an X, Y plane, with horizontal 
axis going to the right as positive Y, and vertical axis going downward 
as X, and the data on it are contiguous in memory, then the access pattern
will be `3 * x + y`, given any X, Y as coordinate. For example, if this is a
2 by 3 grid

```
0, 1, 2
3, 4, 5
```

We can actually represent that in tinygrad:

```python
idxs = [Variable("x", 0, 100), Variable("y", 0, 100)]
e1, e2 = st.expr_idxs(idxs)
print(e1.render())
print(e2.render())
```

e1 render will output `((x*3)+y)` and e2 render will output `1`. e1 result
is straightforward, what does e2 mean? It's used when the shape has masks on it.

Let's say our 2 by 3 has a mask, such that only the first two columns and first
two rows are valid, for a 3 by 3 grid, like the following, and our mask 
argument will state that only the value `0, 1, 3, 4` is valid:

```
0, 1, 2
3, 4, 5
6, 7, 8
```

In code, that looks like:

```python
view = View.create(shape=(x, 3), mask=((0, 2), (0, 2)))
st = ShapeTracker((view,))
```

Now if we render it:

```python
idxs = [Variable("x", 0, 100), Variable("y", 0, 100)]
e1, e2 = st.expr_idxs(idxs)

print(e1.render())
# ((x*3)+y)

print(e2.render())
# ((x<2) and (y<2))
```

You see that e1 is the same, but e2 is now representing the condition for the
range each input is valid. You can modify the mask ad see that it may output
more complex result. For example, if our mask is now `(1, 2), (0, 2)`:

```python
x = Variable("k", 1, 100)
view = View.create(shape=(x, 3), mask=((1, 2), (0, 2)))
st = ShapeTracker((view,))
idxs = [Variable("x", 0, 100), Variable("y", 0, 100)]
e1, e2 = st.expr_idxs(idxs)
print(e1.render())
# (3+y)
print(e2.render())
# (((x*-1)<0) and (x<2) and (y<2))
```

Our mask is stating that only the first row is valid, and within
the first row, only zeroth and first column is valid. As such, the result
no longer depends on x, since there's only one row, hence the access pattern
is just `3 + y`. In addition, x must be 1, or equivalently, `(((x*-1)<0) and (x<2)`
and that y must be less than 2 and greater or equal than zero.


## Variable and symbolic 

By now you might be wondering what these "Variable"s are. They come in handy
when dealing with IR. Suppose we want to render a loop, we will need to know
the start and stop condition. We could design a function that takes in two
inputs:

```python
def render_loop(self, start, stop):
  self.uops.add(UOps.LOOP, dtypes.int32, (self.const(start), self.const(stop)))
```

In reality though, passing start, stop isn't scalable to handle more complex
situation. What is done instead is pass in an object that has start and stop
attribute, and to generalize it, it is an object with `min` and `max` 
attributes. For reference, this is the render_loop definition inside the Linearizer
class, the `const` method that it calls, is also shown:

```python
  def render_loop(self, xx:List[Variable]) -> Tuple[UOp, ...]:
    new_loops = {x.expr:self.uops.add(UOps.LOOP, dtypes.int32, (
      self.const(x.min) if isinstance(x.min, int) else cast(Node, x.min).render(self.render_ops, self),
      self.const(x.max+1) if isinstance(x.max, int) else cast(Node, x.max+1).render(self.render_ops, self)), cachable=False) for x in xx if not isinstance(x, NumNode) and x.expr is not None}  # noqa: E501
    self.loop_uops.update(new_loops)
    return tuple(new_loops.values())

  def const(self, b:ConstType, dtype:DType=dtypes.int32, insert_before=None) -> UOp:
    return self.uops.add(UOps.CONST, dtype, tuple(), b, insert_before=insert_before)
```

It may be easier to understand if we play with it. Consider the
following example, The `c = a.dot(b)` is to satisfy the initializer requirement
that there must be a valid AST, you could have used other operations like `c = a + b`;
in order to run the `render_loop` method, we need to set up the linearizer,
its uops and loop_uops attributes. The set up was done in the `linearize()` method,
but that's a huge implementation, so I'm just showing the necessary steps to
reproduce a loop operation. The important bit is that we set up a variable 
instancee that has min set to 0 and max set to 10, and we use this to convey
the intention that we want to set up a loop from 0 to 10 inclusive.

```python
from tinygrad import Tensor
from tinygrad.engine.schedule import create_schedule
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.shape.symbolic import Variable
from tinygrad.codegen.uops import UOps, UOp, UOpGraph
a = Tensor([1,2,3,4]).realize()
b = Tensor([5,6,7,8]).realize()
c = a.dot(b)
s = create_schedule([c.lazydata])[0]
k = Linearizer(*s.ast)
k.uops = UOpGraph()
k.loop_uops = {}
var = Variable('x', 0, 10)
k.render_loop([var])
k.uops.print()
```

The output is:

```
   0 UOps.CONST          : dtypes.int                []                               0
   1 UOps.CONST          : dtypes.int                []                               11
   2 UOps.LOOP           : dtypes.int                [0, 1]                           None
```

Combining the output with the implementation, You can see that `render_loop` 
takes in the `Variable` instance and find its min value, if it is an integer,
it sets up a CONST op, and keep a reference to it. It does the same
for the max value. And our UOps list is now able to represent a loop
operation.

This is actually sufficient to generate some code:

```python
from tinygrad.renderer.cstyle import uops_to_cstyle, MetalLanguage
print(uops_to_cstyle(MetalLanguage(), 'tester', k.uops))
```

And the output is:

```c++
#include <metal_stdlib>
using namespace metal;
kernel void tester(uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  for (int ridx0 = 0; ridx0 < 11; ridx0++) {
}
```

You notice a problem, the loop block isn't closed, that's because we didn't add
the "ENDLOOP" op, but that's another topic.

Although it may look like all Variable did is just holding the min and max
value, its utility become more obvious when you think of those values
as dynamic. Before the loop is rendered, we may need to check a bunch of things,
for example, whether to unroll the loop, whether to split the loop into smaller
iterations and launch more threads, etc. It is hard to manage if you are passing
two ingeters around, whereas encoding it inside a Variable allows more flexibility.
In fact, if you look into the [implementation](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/symbolic.py),
you will see that it allows you to keep track of all the algebraic operations
on it and finally render the expected value.

In fact, the valid and rendered access pattern, back to our shape tracker example
earlier, were done using these Variable instances (`Node` is the parent class
of `Variable`, and `NumNode` is another class that inherits from `Node`):

```python
def _expr_view(view:View, idxs:List[Node], valid:Optional[Node]=None) -> Tuple[Node, Node]:
  assert len(idxs) == len(view.shape), f"need an idx for all dimensions {idxs} vs {view.shape}"
  iexpr: List[Node] = [NumNode(view.offset) if isinstance(view.offset, int) else view.offset]
  vexpr: List[Node] = [valid] if valid is not None else []
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr.append(idx*st)
    if m is not None: vexpr += [create_ge_node(idx, m[0]), create_lt_node(idx, m[1])]  # idx >= m[0], idx < m[1]
  return Node.sum(iexpr), Node.ands(vexpr)
```

## expand_node

Now that you have a basic understanding of how loops are formed, and what
Variable is, and assuming you know what [loop unrolling (upcasting)](upcast.md) is,
the `expand_node`'s effect and purpose should become clear once you look
at the test cases:

```python
class TestLinearizerHelper(unittest.TestCase):
  def test_num_node_expand(self):
    a = NumNode(42)
    assert expand_node(a) == [a]

  def test_variable_expand(self):
    a = Variable("a", 5, 7)
    assert expand_node(a) == [a]

  def test_variable_expand_expr_none(self):
    a = Variable("_uidx0", 5, 7)
    assert expand_node(a) == [NumNode(5), NumNode(6), NumNode(7)]

  def test_mul_node_expand(self):
    a = Variable("_uidx0", 5, 7)
    m = MulNode(a, 3)
    assert expand_node(m) == [NumNode(15), NumNode(18), NumNode(21)]
```

Putting it in the context of loop unrolling, if we are iterating a loop
from value 5 to 7 inclusive, and we want to unroll it, so it loads the
data from address 5, 6, 7 directly, we call `expand_node`, and then
iterate through each `NumNode` and add a `LOAD` UOp to the UOpGraph. You 
can see that it even allows you to make adjustment to the Variable
by multiplying it with values, and it is able to figure out the correct
address to load from!