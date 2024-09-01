# Build Tinygrad from scratch, in one weekend

> Pre-requisite: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

If you are struggling with bounties, it could be because you are missing some
concepts when it comes to implementing an ML framework, let's fill that gap 
by building ourselves a miniature version of tinygrad from scratch, without
any dependency. You can find the completed code in the "mygrad" folder.

## Chapter 1 - arithmetic engine
At its core, an ML framework enables arithmetic operation on multi dimensional
data, for example, a 2 by 2 matrix can be multiplied with another 2 by 2
matrix, and get the resulting 2 by 2 matrix. We would like to do something 
like below:

```python
a = Tensor([1,2,3])
b = Tensor([1,2,3])
c = a.mul(b) # --> c would have result [1,4,9]
```

The implementation is trivial for now:

```python
class Tensor:
    def __init__(self, data):
        self.data = data

    def mul(self, other):
        result = []
        for i in range(len(self.data)):
            result.append(self.data[i] * other.data[i])
        return Tensor(result)
```

There are a few things we need to address though:

## Generalizing the elementwise pattern

If you were to implement addition, you will notice that it looks very much
the same as the multiplication:

```python
    def add(self, other):
        result = []
        for i in range(len(self.data)):
            result.append(self.data[i] + other.data[i])
        return Tensor(result)
```

As a result, we can extract this to a general method called elementwise. You 
will see later how almost all the operation can be boiled down to "elementwise"
and "reduce" (more on reduce later).

```python
def elementwise(op, tensor_a, tensor_b):
    ops = {
        "add": lambda x, y: x + y,
        "mul": lambda x, y: x * y
    }
    result = []
    for i in range(len(self.data)):
        x = tensor_a[i]
        y = tensor_b[i]
        op_func = ops[op]
        result.append(op_func(x, y))
    return Tensor(result)

class Tensor:
    def add(self, other):
        return elementwise(self, other, "add")
    
    def mul(self, other):
        return elementwise(self, other, "mul")
```

## Unary, binary and ternary operation

Addition and multiplication are binary operation, as they take two
element to produce a single output. Unary operations are those that operates
on a single tensor, for example, log, exponential, negation. There are common
ones you see in day to day ML code and you can see that they can be implemented
similarly as an elementwise op.

```python
def elementwise_unary(op, tensor):
    ops = {
        "log": lambda x: math.log(x),
        "exp": lambda x: math.exp(x),
        "neg": lambda x: -x,
    }
    result = []
    op_func = ops[op]
    for i in range(len(tensor.data)):
        result.append(op_func(tensor.data[i]))
    return Tensor(result)
```

We also have ternary operation, one of them is the `where` op, that was used
in the tril method when constructing a causal attention mask, here's an example
usage:

```python
a = Tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
a.tril() # --> 
# [
#   [1,0,0],
#   [4,5,0],
#   [7,8,9],
# ]
```

The way it works is that a triangular tensor is first formed with its
value filled with either zero (indicating False) or one (indicating True), 
and its valued is checked in an elementwise fashion, if the value is one/True,
then the first tensor's corresponding value will be filled in the result,
otherwise, the result will take the value from the second tensor:
```python
a = Tensor([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
])
b = Tensor([
    [8,8, 8],
    [8,8,8],
    [8,8,8]
])
c = Tensor([
    [9, 9, 9],
    [9, 9, 9],
    [9, 9, 9]
])

a.where(tensor_b, tensor_c)
# Output will be
# [
#     [8,9,9],
#     [8,8,9],
#     [8,8,8]
# ]
```

In fact, this can also be expressed as an elementwise op:

```python
def elementwise_ternary(op, tensor_a, tensor_b, tensor_c):
    ops = {
        "where": lambda condition, x, y: x if condition != 0 else y
    }
    result = []
    op_func = ops[op]
    for i in range(len(tensor.data)):
        result.append(op_func(tensor_a.data[i], tensor_b.data[i], tensor_c.data[i]))
    return Tensor(result)
```

Hopefully you are starting to see that the three categories can be expressed
as a single elementwise function:

```python
ops = {
    "add": lambda i, x, y: x.data[i] + y.data[i],
    "mul": lambda i, x, y: x.data[i] + y.data[i],
    "neg": lambda i, x: -1 * x.data[i],
    "log": lambda i, x: math.log(x.data[i]),
    "where": lambda i, condition, x, y: x.data[i] if condition != 0 else y.data[i]
}
def elementwise(op, *tensors):
    length = len(tensors[0].data)
    result = []
    for i in range(length):
        op_func = ops[op]
        result.append(op_func(i, *tensors))
    return Tensor(result)
```

## Reduce op

Elementwise operation is a 