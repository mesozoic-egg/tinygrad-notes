# Taking a deeper look at the code generation

```python
a = Tensor([1])
b = Tensor([2])
c = a + b
print(c.numpy())
```

will generate the following uops

```
   0 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (0, 'data0', True)
   1 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (1, 'data1', False)
   2 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (2, 'data2', False)
   3 UOps.CONST          : dtypes.int                []                               0
   4 UOps.LOAD           : dtypes.int                [1, 3]                           None
   5 UOps.LOAD           : dtypes.int                [2, 3]                           None
   6 UOps.ALU            : dtypes.int                [4, 5]                           BinaryOps.ADD
   7 UOps.STORE          :                           [0, 3, 6]                        None
```

which then generates:

```c++
#include <metal_stdlib>
using namespace metal;
kernel void E_(device int* data0, const device int* data1, const device int* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int val0 = *(data1+0);
  int val1 = *(data2+0);
  *(data0+0) = (val0+val1);
}
```

# LOAD

a load operation generates this string: `*(data1+0)`. We know that a load
must have an input referrinng to the data, and the other input referring to
the pointer position offset. We have just 1 element, so the offset is zero. 
This is the load renderer:

```python
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    out_val = f"*({buf_name}+{idx})"
    return out_val
```

`idx` will have a value of zero in this case. What if we have more than 1 
elements, then `idx` will have a value of 1, 2, 3, etc. But the implementation
is done via the [[thread_position_in_threadgroup]], so if the input is:

```python
a = Tensor([1, 2, 3])
b = Tensor([2, 5, 6])
c = a + b
print(c.numpy())
```

then idx will have a value of `lidx0`, and the output code is: 

```c++
kernel void E_2(device int* data0, const device int* data1, const device int* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 2 */
  int val0 = *(data1+lidx0);
  int val1 = *(data2+lidx0);
  *(data0+lidx0) = (val0+val1);
}
```

with the uops list being:
```
   0 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (0, 'data0', True)
   1 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (1, 'data1', False)
   2 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (2, 'data2', False)
   3 UOps.SPECIAL        : dtypes.int                []                               (0, 'lidx0', 2)
   4 UOps.LOAD           : dtypes.int                [1, 3]                           None
   5 UOps.LOAD           : dtypes.int                [2, 3]                           None
   6 UOps.ALU            : dtypes.int                [4, 5]                           BinaryOps.ADD
   7 UOps.STORE          :                           [0, 3, 6]                        None
```

You see that idx with value zero came from previously the `CONST` op, now it is a `SPECIAL` op.

We know conceptualy idx takes the value from the second input, but when code gen
encounters the LOAD op, how does it go and grab the value from the input? Inside
`uops_to_cstyle`:

```python
      elif uop is UOps.LOAD:
        print(r[vin[0]]) # Added for illustration
        print(r[vin[1]]) # Added for illustration
        val = lang.render_load(dtype, r[vin[0]], vin[0].dtype, r[vin[1]], vin[0].uop is UOps.DEFINE_LOCAL)
```

the datainput is straightforward, the uop that maps to the first input: `r[vin[0]]`
the idx came from here: `r[vin[1]]`. So the "Special" doesn't have any effect here,
it is literally just grabbing the argument of the op. If it's a CONST op, value
will be zero, if it's a SPECIAL, value will be "lidx0". You can check the print
statement and run it yourself.

You may be confused about the r variable, it is initially an empty dictionary,
and gets populated with content as we iterate through the uops list item one by
one, among them the CONST and SPECIAL op. 

In the single element case, a CONST op is first encountered and gets handled
as such:

```python
      elif uop is UOps.CONST: 
         r[u] = lang.render_const(args, dtype) if args >= 0 else f"({lang.render_const(args, dtype)})"
```

with the render_cosnt function being
```python
  def render_const(self, x:ConstType, dtype:DType) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    elif dtype == dtypes.float64: val = f"{x}"
    else: val = f"{x}f" if dtypes.is_float(dtype) else f"{x}" if dtypes.is_int(dtype) else f"{x}".lower()
    return (self.render_cast([val] * dtype.count, dtype) if dtype.count > 1 or dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)
```

which in the most simple case, is just the value being printed either with the 'f'
suffix or as a plain number. And our dictionary now becomes

```python
{
   CONST: 0
}
```

In the case of the SPECIAL op, 

```python
      elif uop is UOps.SPECIAL:
        kk(f"int {args[1]} = {lang.code_for_workitem[args[1][0]](args[0])}; /* {args[2]} */")
        r[u] = args[1]
```

with args being `(0, 'lidx0', 2)`.

You can now see where the value of idx came from in render load, it's the input value
that's being added to the `r` dictionary. If it's a multi element, we use a 
SPECIAL op to represent the pointer arithmetic, and store the number of thread
within threadgroup to iterate through the list. If it's a single element,
we store the value of a CONST directly as input and put it in the `r` dictionary,
so it renders as `*(data1+0)`.

# CAST

another common operation is type casting. We may start off with some integer
but want the output to be a float.

```python
from tinygrad import Tensor, dtypes

a = Tensor([1, 3])
b = Tensor([4, 3])
c = (a + b).cast(dtypes.float32)
print(c.numpy())
```

the metal code is 
```c++
  int lidx0 = lid.x; /* 2 */
  int val0 = *(data1+lidx0);
  int val1 = *(data2+lidx0);
  *(data0+lidx0) = (float)((val0+val1));
```

THe only change is just the `(float)` part, how is this rendered? Let's see
uops:

```
   0 UOps.DEFINE_GLOBAL  : ptr.dtypes.float          []                               (0, 'data0', True)
   1 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (1, 'data1', False)
   2 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (2, 'data2', False)
   3 UOps.SPECIAL        : dtypes.int                []                               (0, 'lidx0', 2)
   4 UOps.LOAD           : dtypes.int                [1, 3]                           None
   5 UOps.LOAD           : dtypes.int                [2, 3]                           None
   6 UOps.ALU            : dtypes.int                [4, 5]                           BinaryOps.ADD
   7 UOps.CAST           : dtypes.float              [6]                              dtypes.float
   8 UOps.STORE          :                           [0, 3, 7]                        None
```

step 7 introduces the cast operation that operates on step 6. Let's see
the code gen on how it handles CAST:

```python
      elif uop in {UOps.CAST, UOps.BITCAST}:
        if uop is UOps.BITCAST:
          assert len(vin) == 1
          precast = ssa('precast')
          kk(f"{lang.render_dtype(cast(DType, vin[0].dtype))} {precast} = {r[vin[0]]};")
          val = lang.render_cast([precast], dtype, bitcast=True)
        else:
          val = lang.render_cast([r[x] for x in vin], dtype, bitcast=False)
```

In our case, it's just calling `render_cast` with the input value, which is
stored in the register `r` with the value of the addition operation. dtype 
would be float as we wanted to cast it to float. Let's see `render_cast`:

```python
  def render_cast(self, x:List[str], var_dtype:DType, bitcast=False) -> str:
    if bitcast: return f"(*(({self.buffer_prefix}{self.render_dtype(var_dtype)}*)&{x[0]}))"
    if len(x) == 1: return f"({self.render_dtype(var_dtype)})({x[0]})"
    assert len(x) == var_dtype.count, f"cast is wrong size {len(x)} != {var_dtype.count}"
    assert self.float4 is not None, "vectorized cast is not supported on this platform"
    return f"{self.float4.replace('float4', self.render_dtype(var_dtype))}({','.join(x)})"
```

Our scenario hits line 2, so we return `(float)((val0+val1))`. Note that
the value of x would be `["(val0+val1)"]`. 

Also note that what I show here is the parent method defined by `CStyleLanguage`
that may not reflect the actual GPU usage. In metal that's actually the case,
so our MetalLanguage, which inherits from CStyleLanguage, overrides the render_cast
method:

```python
  def render_cast(self, x: List[str], var_dtype: DType, bitcast=False) -> str:
    return f"as_type<{self.render_dtype(var_dtype)}>({x[0]})" if bitcast else super().render_cast(x, var_dtype)
```

Although our simple case it falls back to the parent method, it will use the `as_type`
keyword for something unique to metal in certain scenarios. This is part of how you
would extend tinygrad to custom accelerator.

# ALU

All the arithmetic logic unit are handled similarly. In our case it is
an addition, rendered as `(val0 + val1)`.

```python
      elif uop is UOps.ALU:
        # remove parens if ALU types are the same. TODO: can do more here
        if args in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: 
         operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in vin]
        else: 
         operands = [r[v] for v in vin]
        val = lang.code_for_op[args](*operands, dtype)
        assert child_count[u] != 0, f"childless ALU op found {u}"
        # TODO: fix index rendering issue. fix clang nested max macro issue
        if child_count[u] <= 1 and args is not BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val
        else: kk(f"{lang.render_dtype(dtype)} {ssa('alu',u)} = {val};")
```

In our case, `operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in vin]`
is executed. Recall that vin is the two load operation, and the operands value
end up being `['val0', 'val1']` given how the two loads stored the value 
in register. `args` is 'ADD'. 

code_for_op is defined in the shared CStyleLanguage class unless overriden by
specific inheritants:

```python
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype is dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
    UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})",
    BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPEQ: lambda a,b,dtype: f"({a}=={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"}
```

and an ADD operation is just adding concatenating the two operands with
a plus sign and surround them with parentheses. Finally we store the
rendered result in the register dictionary ` r[u] = val` such that
the previous CAST operation will pick it up. Similarly you can see
how other operations are implemented.

