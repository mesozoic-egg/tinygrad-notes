# Documentation on Tinygrad's IR

Tinygrad operates on fundamental primitive operations called Uops, categorized as follows:

1. ALU: Models hardware capabilities to manipulate tensor combinations.
2. Movement: Defines data sharing between devices, resolves data dependencies for ALU operations.
3. Directives: Models decision points in the code, controlling execution.

These building blocks enable op-wise backpropagation and gradient calculations. Complex functions, like a Sigmoid activation layer, can be expressed using these ops. Ops remain device-agnostic, retaining the properties of their assigned datatypes. Lazy realization optimizes scheduling based on hardware constraints and operational feedback.

When performing any operation on a tensor, such as addition:

```python
a = Tensor([1], dtype=int)
b = Tensor([2], dtype=int)
c = a + b
c.realize()
```

This automatically executes the ALU operation for addition, defined in tensor.py's `__add__`, storing the result on the default device. Graphs can be visualized using graphing tools.

We can construct elements of a graph by hand in this way:

```python
from tinygrad.codegen.uops import UOpGraph, UOps
g = UOpGraph()
c0 = g.add(UOps.CONST, dtypes.int, arg=0)
```

and can be rendered into target platform's code as such:

```python
s = uops_to_cstyle(MetalLanguage(), 'tester', g)
```

The main way to modify the graph is by calling the `add` method with the 
following arguments:

- `dtype`: this specifies the data type. E.g. `dtypes.int`, `dtypes.float`
- `vin`: this specifies the input to the UOp. For example, in a multiplication
operation, the `vin` would be a tuple containing the two input operands.
- `arg`: this is the argument to the UOp. Its content is specific to each
uop and is utilized during the code generation process. For example,
a global variable may specify a tuple, with one of the elements being
the name of the variable, then the code generation process will extract
that element and use it to generate the variable name.

## CONST

Uops.CONST declares a constant variable. There are two required parameters 
in `add`: `dtype` and `arg`.

Example:
```python
c0 = g.add(UOps.CONST, dtypes.int, arg=10)
```

`c0` can then be used as inputs for other UOp.

## DEFINE_GLOBAL

UOps.DEFINE_GLOBAL declares a global variable. It is used as the parameter
list for the function. 

- `arg`: a three element tuple:
  - `0`: the index of the parameter in the parameter list
  - `1`: the name of the parameter
  - `2`: whether the parameter is mutable

- `vin`: omitted or an empty tuple

For example, when declaring a parameter that will be passed to the kernel:

```python
c1 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(1, "data0", True)
```

## LOOP and ENDLOOP

As its name suggests, they set up loop.

LOOP:
- `vin`:
  - `0`: The start value of the loop (must be CONST UOp)
  - `1`: end value (must be CONST UOp)

ENDLOOP:
- `vin`:
  - `0`: The loop UOp
Example:

```python
c0 = g.add(UOps.CONST, dtypes.int, arg=0)
c1 = g.add(UOps.CONST, dtypes.int, arg=10)
loop = g.add(UOps.LOOP, dtype=dtypes.int, vin=(c0, c1))
endloop = g.add(UOps.ENDLOOP, vin=(loop,))
```

The rendered loop looks like this:

```c++
for (int ridx0 = 0; ridx0 < 10; ridx0++) {
}
```

## STORE

STORE is for writing value to the output, which comes in the form of a parameter
passed to the kernel function.

- `dtype`: None
- `vin`: None
- `arg`: Three values must be UOp instance
  - `0`: the UOp for the output
  - `1`: the index position in the output to store the value in 
  - `2`: the value to store

Example
```python
c1 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(0, "data0", True))
c2 = g.add(UOps.CONST, dtype=dtypes.int, arg=0)
c3 = g.add(UOps.CONST, dtype=dtypes.int, arg=10)
store = g.add(UOps.STORE, vin=(c1, c2, c3))
```

and it will render:

```c++
kernel void tester(constant int& data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  *(data0+0) = 10;
}
```

## LOAD

This allows for indexing a value from the input given the offset

- `vin`:
  - `0`: The input value
  - `1`: The offset

Example (see the ALU example for the generated code):
```python
input_value = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(2, "data2", False))
position = g.add(UOps.CONST, dtype=dtypes.int, arg=0)
loaded = g.add(UOps.LOAD, dtype=dtypes.int, vin=(input_value, position))
```

## ALU

ALU is for arithmetic, logical, and bitwise operations.

- `vin`:
  - `0`: the zeroth operand
  - `1`: the first operand

- `arg`:
  - `0`: the operation type

It is usually used in conjunction with other ops, for example, to load the first
element from two input arrays and add them together:

```python
c1 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(0, "data0", True))
x1 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(1, "data1", False))
x2 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(2, "data2", False))
pos_input = g.add(UOps.CONST, dtype=dtypes.int, arg=0)
x1_loaded = g.add(UOps.LOAD, dtype=dtypes.int, vin=(x1, pos_input))
x2_loaded = g.add(UOps.LOAD, dtype=dtypes.int, vin=(x2, pos_input))
c4 = g.add(UOps.ALU, dtype=dtypes.int, vin=(x1_loaded, x2_loaded), arg=BinaryOps.ADD)
pos = g.add(UOps.CONST, dtype=dtypes.int, arg=0)
store = g.add(UOps.STORE, vin=(c1, pos, c4))
```

## Special

GPU kernels are usually executed in SIMT fashion, meaning each thread will need
to identify itself among all the other threads, such that it can fetch the correct
data. In the ALU example above, we are explicitly fetching the zeroth element
via the CONST UOp, but we might want to declare a UOp that fetches element
based on the threadID.

- `arg`:
  - `0`: incremental index among all the special uop
  - `1`: name of the index
  - `2`: Upper limit (exclusive)

Example:
```python
position = g.add(UOps.SPECIAL, dtype=dtypes.int, arg=(0, "gidx0", 10))
```

This means the thread is launched in a group containing ten threads, and each
thread will get the value by iterating from 0 to 10 (exclusive). We can now
modify the ALU example:

```python
c1 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(0, "data0", True))
x1 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(1, "data1", False))
x2 = g.add(UOps.DEFINE_GLOBAL, dtype=dtypes.int, vin=(), arg=(2, "data2", False))
pos_input = g.add(UOps.SPECIAL, dtype=dtypes.int, arg=(0, "gidx0", 10))
x1_loaded = g.add(UOps.LOAD, dtype=dtypes.int, vin=(x1, pos_input))
x2_loaded = g.add(UOps.LOAD, dtype=dtypes.int, vin=(x2, pos_input))
c4 = g.add(UOps.ALU, dtype=dtypes.int, vin=(x1_loaded, x2_loaded), arg=BinaryOps.ADD)
pos = g.add(UOps.CONST, dtype=dtypes.int, arg=0)
store = g.add(UOps.STORE, vin=(c1, pos, c4))
```

and the generated code becomes:

```c++
kernel void tester(constant int& data0, constant int& data1, constant int& data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 10 */
  int val0 = *(data1+gidx0);
  int val1 = *(data2+gidx0);
  *(data0+0) = (val0+val1);
}
```
