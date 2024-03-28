# How dimension merging works

There's a function frequently invoked: `_merge_dims`. For example, at the end
of my last [post](shapetracker.md), there are two views that are created when
View.create method decides the new view isn't compatible with the old one. 
This block of code was invoked:

```python
    for merged_dim, new_stride, real_dim in reversed(_merge_dims(self.shape, self.strides, self.mask)):
      acc = 1
      # TODO: this <= and != is for symbolic!?
      while acc <= merged_dim and acc != merged_dim and (new_dim := next(r_new_shape, None)):
        strides.append(new_stride)
        if new_dim != 1: new_stride *= (new_dim if (acc :=  acc * new_dim) < real_dim else 0)
      if acc != merged_dim: break
    #...
```

First look at what _merge_dims does, it's a short implementation.

```python
def _merge_dims(shape:Tuple[int, ...], strides:Tuple[int, ...], mask:Optional[Tuple[Tuple[int, int], ...]]=None) -> Tuple[Tuple[int, int, int], ...]:
  # merge contiguous subparts or zero strided dims. ret = List[(merged_dims, stride, merged dims w/o zero stride), ...]
  if not shape: return tuple()
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0], shape[0] if strides[0] else 0)]
  # wrt merging zero strided dimensions
  merging = strides[0] == 0 and (mask[0][1] - mask[0][0] == 1 if mask else shape[0] == 1)
  for i, (sh, st) in enumerate(zip(shape[1:], strides[1:]), start=1):
    if sh == 1: continue
    if merging or ret[-1][1] == sh * st: # mergeable
      ret[-1] = (ret[-1][0] * sh, st, (sh if merging else ret[-1][2] * sh) if st else 0)
    else: ret.append((sh, st, sh if st else 0)) # begin new
    # merging ends with either non-zero strided dim or zero strided dim with mask range > 1
    merging = st == 0 and (mask[i][1] - mask[i][0] == 1 if mask else sh == 1)
  return tuple(ret)
```

On a high level, it attempts to convert a view that represent multidimensional data 
into something 2D as possible. Simplest example is if you have the 8 letters
being represented as a cube:

```
[a,b,c,d,e,f,g,h] --> linear memory layout from memory address 0 to 7


[
  [
    [a,b],
    [c,d]
  ],
  [
    [e,f],
    [g,h]
  ]
] --> with a shapetracker, we can define access pattern such that users will 
think the data actually look like this, whereas the data is still linear
```

This is how to create such a shapetracker and I will explore some of its property

```python
from tinygrad.tensor.shape import ShapeTracker

a = ShapeTracker.from_shape((2,2,2))
print(a.expr_idxs()[0].render()) # --> ((idx0*4)+(idx1*2)+idx2)
                                 # refer to my previous post to interpret the output

# The actual stride and mask info is on the last (and only for now) view
print(a.views[-1].shape) # --> (2,2,2)
print(a.views[-1].strides) # --> (4,2,1)
print(a.views[-1].mask) # --> None
```

So we see that in order to represent 8 letters linearly laid out on a strip, 
we need two pieces of info, shape (2,2,2) and strides (4,2,1). Do read up
on the concept of strides if you are unfamiliar with it. What _merge_dims does
is it takes the two pices of info and reduce it down (ignore mask for now)

```python
from tinygrad.shape.view import _merge_dims
shape = (2,2,2)
strides = (4,2,1)
print(_merge_dims(shape, strides))
# ---> ((8, 1, 8),)
```

The output is a tuple with a single element. This element represent the zeroth
dimension, in our case, also the only dimension, indicating the merge process
has merged everything into a single dimension. This element in turn has three
items in it, the first is the shape, the second is the stride, the third I will
ignore for now. The overall output says, given your input of a cube, I have
reduced it to a single dimension with 8 elements, with each element spaced 1
unit apart from each other. In other words, the elements can be represented 
contiguously on a linear memory layout.

What if we have something that was broadcasted? If you are unfamiliar with broadcasting,
[some reading might help](https://www.google.com/search?q=broadcasting+pytorch).

```python
a = ShapeTracker.from_shape((2,)) # We start with a two element shape [a,b]

# We want to broadcast it into a 2 X 2 X 2 cube

a = a.reshape((1,1,2)) # First, we increase the number of dimension by reshaping it into a 1 by 1 by 2 cube
                       # the 1 by 1 part is required, if you reshape it as (2,2,2) it won't work
                       # due to how some shape management requirements I haven't discussed yet
a = a.expand((2,2,2)) # Then, we expand it into a 2 by 2 by 2 cube
print(a.expr_idxs()[0].render()) # idx2
print(a.views[-1].shape) # (2, 2, 2)
print(a.views[-1].strides) # (0, 0, 1)
print(a.views[-1].mask) # None
```

The render output (idx2) says to get the value of the cube, you take just the number
of the second dimension. So if you want the element at [0,0,1], it's at memory position 1.
If you want element at [1,1,0], it's at memory position 0. Remember we only have two
elements in memory, and this cube just "broadcast" into these two elements. Such an
output is essentially encoded by the stride (0,0,1), which says any movement you take
in the zeroth and first dimension will not change the memory access location, and 
any single step/movement you take in the second dimension will increment the memory
location by 1. Now we pass the shape and stride into _merge_dims

The cube looks like this. I use capital A and B to indicate the actu
```
[
  [
    [a,b],
    [a,b],
  ],
  [
    [a,b],
    [a,b]
  ]
]
```

```python
shape = (2,2,2)
strides = (0,0,1)
print(_merge_dims(shape, strides))
# --> ((4, 0, 0), (2, 1, 2))
```

The output says, the merged output has two dimensions. The first dimension
has a length of 4, stride 0, second dimension has a length of 2 and stride 1 
(refer back to the earlier paragraphs for explanation). Which essentially means
the best we can do is to reduce the cube into a grid:

```
[
  [a,b]
  [a,b]
  [a,b]
  [a,b]
]
```

Do some indexing operation with stride to see how each element maps to the 
two element memory layout.

Why are there two dimensions instead of just one like in the first example?
Because merging dimension has to preseve the information on individual elements while
also reducing complexity. Reducing complexity part is straightforward. Think
of adding a number to each element in a cube,
you would need three for loops, but if you reduce it to a grid, there's only
2 for loops. Even with a naive iterative approach on a simple case you have already 
simplified things a great deal, think of how much more gain you get if you reduced a
10 dimensional structure into a grid and with other optimziation in place.

Preserving information may be less obvious. Think about back propagation. If the
element a and b are just some bias you add to a binomial output and there are
thousands of nodes referencing them. When you do backprop, you need to add up
all the gradient in each node:

```
[
  [a,b]
  [node1 referencing a, node 2 reference b]
  [node3 referencing a, node4 referencing b],
  [node5 referencing a, node6 referencing b],
  ...
]
```

If you turn the whole thing into just two elements, you have no room to keep
the nodes. 

I like to think of it as just a rule, merging dimensions can't reduce
the number of elements, you end up having just as many elements as you started 
with, even if they are broadcasted elements. So if you have 8 elements in broadcasted
space, and there are 2 in memory backing them, the best you can do is a grid. If 
you have 8 elements backed by exactly those 8 in memory, then you can reduce to a linear
strip.

Next I want to talk about the third element, it's called `real_dim`. It refers
to the number of concrete element in memory backing this dimension. Recall in
the first example

```
shape = (2,2,2)
strides = (4,2,1)
print(_merge_dims(shape, strides))
# ---> ((8, 1, 8),)
```

We reduced an 8 element cube into a single dimension, and we know that
there are 8 elements residing in memory, hence the thrid element is 8.

In our second example with the broadcasted cube

```
shape = (2,2,2)
strides = (0,0,1) 
print(_merge_dims(shape, strides))
# ---> ((4, 0, 0), (2, 1, 2))
```

Recall the end result is a grid. The moving along the zeroth dimension refers to 
walking in the column direction (up and down), moving along the first dimension
refers to the row. We know that it doesn't matter how you move along the column,
you are always accessing the same element in memory. We describe this as having
stride of zero. We can also come up with another way to describe this, which is that
there's no concrete element on this dimension, and we use zero to denote it. In the
first dimension, we know that there are two elements that are actually concrete,
so the real_dim is 2.

The actual implementation details are covered in this [PR](https://github.com/tinygrad/tinygrad/pull/2218/files#diff-9c4bc25c609588862d30e9206f30ec2c4ca1cd1de74da0865124af72e184b42d)
with an accommpanying doc that explains the math and proofs behind. Honestly
I still don't fully understand it, and would appreciate some help.
