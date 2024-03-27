# How ShapeTracker works

Shapetracker enables zero-cost movement operations. What that means is you need a way to represent linearly laid out data on the memory strip as a multi dimensional structure, and also that if you have to reshape a huge tensor, the underlying data that's in the memory doesn't have to changed, it's only the information on how to access those data changes. Let me explain with a concrete example, if you have 8 letters `[a,b,c,d,e,f,g,h]` and throughout your program, you need to present them to the user in various ways, maybe one after another: 1. as a 8 digits laid out linearly. 2. as a 4 by 2 matrix. 3. as a 2 X 2 X 2 cube. So the output would be like this

```
[a,b,c,d,e,f,g,h] --> memory position: [0,1,2,3,4,5,6,7]
```

```
4 X 2 grid
[[a,b,c,d],
[e,f,g,h]]
```

```
2 X 2 X 2 cube
[
  [
    [a,b],
    [c,d]
  ],
  [
    [e,f],
    [g,h]
  ]

]
```
You can't store such cubic structure in memory directly, they will be laid out flat linearly, so you need to come up with a way to access it as if they are a cube or a grid. Also, if your data is large, changing the underlying storage on memory may be expensive, but it turns out the data can remain unchanged (always as [a,b,c,d,e,f,g,h]), we have to just keep a record of how to access them when the data need to be presented (keeping such a record does come with overhead, so it only make sense when your data is large, like thousands of numbers stored in memory).

Tinygrad has a module called shapetracker.py that does this.

```python
from tinygrad.shape.shapetracker import ShapeTracker

# To create a view
v = ShapeTracker.from_shape((8,))

# To get the accessing instruction
print(v.expr_idxs()[0].render()) # --> idx0
```

Ignore the odd syntax for now (they serve a separate purpose I will cover later). The `expr_idxs` rendered result is `idx0`, it means the data has only 1 dimension, and you access the data as is. If you want to access the 3rd element, you just index into it directly: `data[3]`

In other words, this is just a list, and accessing 3rd element is simply just data[3]


```python
v = v.reshape((2, 4))
print(v.expr_idxs()[0].render()) # --> ((idx0*4)+idx1)
```

Now we represent the data as a 2 by 4 grid. And the output `((idx0*4)+idx1)` means if you want to access the 1st row (index 1) and 2nd column (index 2), your target is residing on 1 multiplied by 4 and added by 2. 1 * 4 + 2 is 6, the 6th letter is "f". Now we can make sense of the output better, idx is the dimension, and the suffixed number indicate which dimension it is. 

In other words, if your data is accessed as `grid[1][2]` in the program, then it will find the element in memory with `data[1 * 4 + 2]` (`data[2]`)

Let's do one more:

```python
v = v.reshape((2,2,2))
print(v.expr_idxs()[0].render())  # --> ((idx0*4)+(idx1*2)+idx2)
```

Here, indexing into our data with [1,0,1] will mean accessing the element at position `1 * 4 + 0 * 2 + 1` equal 5. 
`cube[1][0][1]` will be translated to `data[1 * 4 + 0 * 2 + 1]` (`[data[5]`)

So that covers how to represent higher dimensional tensor when the underlying data is a flat list. A second problem it solves is the zero cost movement part. If you have a grid that's 2 by 4, and you want to transpose it to 4 by 2, that means you might have to reorder the data, but reordering data could be expensive if you have millions of elements, and if it turns out you are gonna transpose it back at the end, maybe we can come up with a way to just record how to access a few element as if it's already transposed. That's where stride comes in. 

```python
# Let's start from fresh
v = ShapeTracker.from_shape((4,2))
print(v.expr_idxs()[0].render()) # --> "idx0 * 2 + idx1", we saw this already
v = v.permute((1,0)) # This is the syntax for swapping row and column, it reads: put dimension 1 to position 0, and put dimension 0 to position 1
print(v.expr_idxs()[0].render()) # --> idx1 * 2 + idx0
```

After permuting it, we see the access pattern changed, and if you are accessing the data as `permuted_grid[2,3]` then the data will be at `data[3 * 2 + 2]` (`data[8]`). Whereas previously it would be `data[2 * 2 + 3]`, and that's indeed what happens when you swap the column and row!
```
[
  [a,b],
  [c,d],
  [e,f],
  [g,h]
]
```

is transposed to:

```
[
  [a,c,e,g],
  [b,d,f,h]
]
```

You can try out many more examples, such as expand, shrink, etc. and explore the generated expression. 

Next I want to cover how this works internally

```python
# When you have a shapetracker instance like below
shape = ShapeTracker.from_shape((4,2))
shape = shape.reshape((2,2,2))
shape = shape.reshape((2,4))

# It has a views attributes. It contains a list of `View` objects. The view object stores info on how to access data. For example:

print(shape.views) # (View(shape=(2, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),)

# The output says the view is a 2 by 4 grid, and the strides is used to render the access pattern we explored above
# Here, the strides is (4,1), meaning the data will be stored at idx0 * 4 + idx1. That's essentially how "strides" works btw.

# But I want to show you something more interesting
# If you modify it in a special way, you will see two views
shape = shape.permute((1,0))

print(shape.views) # --> (View(shape=(4, 2), strides=(1, 4), offset=0, mask=None, contiguous=False),)

print(shape.expr_idxs()[0].render()) # --> idx1 * 4 + idx0, nothing special yet

shape = shape.reshape((2,4)) # Now, behold

print(shape.views) # (
#                       View(shape=(4, 2), strides=(1, 4), offset=0, mask=None, contiguous=False), 
#                       View(shape=(2, 4), strides=(4, 1), offset=0, mask=None, contiguous=True)
#                    )

print(shape.expr_idxs()[0].render()) # (((idx1%2)*4)+(idx0*2)+(idx1//2))
```

We noticed that after permute and reshape, there are two views. Having two views means the two are not 
compatible. Such compatibility affects downstream optimization. Now, if you examine the shapetracker's
commonly accessed property like shape and strides

```python
  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()
```

You see that it is just getting the information from its last view. 