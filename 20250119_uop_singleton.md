# UOp is singleton

If you look at the definition of UOp, you will notice that it follows a singleton pattern:

```python
class UOpMetaClass(type):
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:tuple[UOp,...]=tuple(), arg:Any=None, _buffer:Buffer|None=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = ref = weakref.ref(created:=super().__call__(*key))
```

If the four arguments: `op`, `dtype`, `src`, `arg` are the same, then the same class instance will be returned, instead of 
being created. This means you can do comparison directly on two UOp tree:

```python
from tinygrad.ops import UOp, Ops
const1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const2 = UOp(Ops.CONST, dtypes.float, arg=0.5)
print(const1 == const2) # True
```

We can compare the tree also:

```python
const1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const2 = UOp(Ops.CONST, dtypes.float, arg=0.5)
buf1 = UOp(Ops.DEFINE_GLOBAL, arg=1)
buf2 = UOp(Ops.DEFINE_GLOBAL, arg=2)
a = UOp(Ops.ADD, src=(const1, buf1))
b = UOp(Ops.ADD, src=(const1, buf1))
c = UOp(Ops.ADD, src=(const1, buf2))
print(a == b) # True
print(a == c) # False
print(b == c) # False
```

Note that the UOp usage here are just made up. The actual UOp tinygrad generated are more complex and have more rules.

## Checking if two uop trees are almost equal

Singleton pattern makes it easy to modify, transform and compare the AST. For example if you want to check if your trees
are "almost equal". We can see that `a` and `c` only differs in their `Ops.DEFINE_GLOBAL`. You can write a function
that removes `DEFINE_GLOBAL` and compare the rest of the tree:

```python
def remove_buf(uop: UOp):
  src = [remove_buf(_uop)  for _uop in uop.src]
  src = tuple([_uop for _uop in src if _uop is not None])
  if uop.op == Ops.BUFFER: return None
  return uop.replace(src=src)

_a = remove_buf(a)
_c = remove_buf(c)

print(_a)
"""
UOp(Ops.ADD, dtypes.void, arg=None, src=(
  UOp(Ops.CONST, dtypes.float, arg=0.5, src=()),)) 
"""

print(_c)
"""
UOp(Ops.ADD, dtypes.void, arg=None, src=(
  UOp(Ops.CONST, dtypes.float, arg=0.5, src=()),))
"""

print(_a == _c) # True
```