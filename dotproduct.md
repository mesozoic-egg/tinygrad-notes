# Kernel Fusion: How dot product operation works

I will use a simple dot product operation to dissect some internals of tinygrad.
the dot product between [1,2] and [3,4] is 1 * 3 + 2 * 4 = 11. In tinygrad this 
will be

```python
from tinygrad.tensor import Tensor
a = Tensor([1,2])
b = Tensor([3,4])
res = a.dot(b).numpy()
print(res) # 11
```

You can save this script as script.py and run it with the following flags

```bash
DEBUG=5 NOOPT=1 python script.py
```

The DEBUG flag set to 5 will make it output a ton of useful information that allows
us to see the internals. Setting NOOPT to 1 disable optimization so it's easier
to reason about for learning purpose.

There are many sections that are printed. The first we look at is the generated
kernel code (at the end of the output)

```
#include <metal_stdlib>
using namespace metal;
kernel void r_2(device int* data0, const device int* data1, const device int* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    int val0 = *(data1+ridx0);
    int val1 = *(data2+ridx0);
    acc0 = ((val0*val1)+acc0);
  }
  *(data0+0) = acc0;
}
```

I'm using a macbook, so the code is written in Metal (mac's GPU API). The function
that's defined is called a GPU kernel and can run in parallel. So imagine there
are thousands of the above kernel running in parallel independently. In our simple
case though, there's only 1 kernel being launched. Let's dissect the above (it 
requires some understanding of C++ and GPU coding):

`device int* data0`: this is the output data pointer, we store the output by
writing to it like `*(data0+0) = acc0`. Since we have just one kernel, it writes
to a memory location hard coded already, but if there are more than one kernel,
the `+0` may be replaced with something more dynamic.

`const device int* data1`: this is the memory pointer for our first tensor [1,2].
so to access number 1, you would do `*(data1+0)` and for number 2, it is `*(data1+1)`.

`const device int* data2`: similar to data1, but for tensor [3,4]

`uint3 gid [[threadgroup_position_in_grid]]`: this is referring to where the kernel
is in the GPU. Think of the GPU as a grid and has a dimension 4 by 8, then you
can launch 32 (4*8) of the above kernel, and each kernel's gid would be a struct 
with x corresponding to the x axis position, and y being the y axis vertical 
position. So the first kenerl would have gid.x of value 0 and and gid.y of value 0.

`uint3 lid [[thread_position_in_threadgroup]]`: similar to the gid but an extra
layer of organization. The [apple documentation explains it well](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups)

The body part of it is more straightforward, it initialize a 0, and loop through
each element of the two tensors, add them and return the value. In python translation
that would be

```python
def kernel_r_2(data0, data1, data2, gid, lid):
  acc0 = 0
  for i in range(2):
    val0 = data1[i]
    val1 = data2[i]
    acc0 = val0 * val1 + acc0
  data0[0] = acc0
```

How are these code generated? When you perform the .dot operation, an Abstract
Syntax Tree is generated, and it's then converted into a series of linear operations, 
and then a codegen utility takes those linear operations into the actual code.

The AST looks like this
```
  0 ━┳ STORE MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)))
  1  ┗━┳ SUM (0,)
  2    ┗━┳ MUL
  3      ┣━━ LOAD MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)))
  4      ┗━━ LOAD MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)))
```

The zeroth line is universal across all the operation you do, it means storing the computed result somewhere.
At the bottom we see two LOAD, indicating that we are taking in two tensors, and the two are multiplied and
summed across the zeroth dimension. Having this alone isn't sufficient for code gen, because writing out
the code is a linear process, so we want to convert them into a format more friendly for writing code
generator. Hence the below linear ops:

```
step  Op_name               type                      input                           arg
   0 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (0, 'data0', True)
   1 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (1, 'data1', False)
   2 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (2, 'data2', False)
   3 UOps.DEFINE_ACC     : dtypes.int                []                               0
   4 UOps.CONST          : dtypes.int                []                               0
   5 UOps.CONST          : dtypes.int                []                               2
   6 UOps.LOOP           : dtypes.int                [4, 5]                           None
   7 UOps.LOAD           : dtypes.int                [1, 6]                           None
   8 UOps.LOAD           : dtypes.int                [2, 6]                           None
   9 UOps.ALU            : dtypes.int                [7, 8]                           BinaryOps.MUL
  10 UOps.ALU            : dtypes.int                [9, 3]                           BinaryOps.ADD
  11 UOps.PHI            : dtypes.int                [3, 10, 6]                       None
  12 UOps.ENDLOOP        :                           [6]                              None
  13 UOps.STORE          :                           [0, 4, 11]                       None
```

You read the ops from top to bottom and it matches how you would read the generated metal code.
the DEFINE_GLOBAL at position zero refers to data0, our output. The two that follow are for 
input tensors [1,2] and [3,4]. We define an accumulator with value 0 [3 DEFINE_ACC], initialize a
for loop variable with value 0 [4 CONST] that ends upon value 2 [5 CONST]. Then
enters a loop [6 LOOP] and load the value of the first tensor [7 LOAD] and second
tensor [8 LOAD], multiply the two together [9 ALU] and add to the accumulator [10 ALU].
The next operation is PHI, it relates to Single Static Assignment. Tbh I don't understand it
well enough, but the general idea is that during the loop (which ran twice) we actually
created 2 extra accumulator in the eyes of the GPU. Although the metal code we wrote
allow us to overwrite them but this isn't always the case with other GPU programming
languages. So having a PHI is instructing the code gen to figure out which of the
three accumulator (the one initialized at [3 DEFINE_ACC], plus the two that
are created at [10 ALU] within the loop that ran twice) to use for the final value. 
Finally we end the loop [12 ENDLOOP] and write to the result [13 STORE].

The part of the code that generates these is [here](https://github.com/tinygrad/tinygrad/blob/6c7df1445b287131862a628937e03e336c895c0c/tinygrad/codegen/linearizer.py#L171)

And the code gen for the metal platform is [here](https://github.com/tinygrad/tinygrad/blob/6c7df1445b287131862a628937e03e336c895c0c/tinygrad/renderer/cstyle.py#L91)