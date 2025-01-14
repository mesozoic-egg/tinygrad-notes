# Memoryview in tinygrad

Memoryview allows you to handle raw binary data in python style. Depending on your expectation, this could either be
convenient or confusing. In either case, it enables a smooth transit for between different hardware backends. A common
use case is transfering data from numpy to GPU, then back to terminal for printing the result.

Let's see some basic stuff first.

```python
a = memoryview(b"abc")
print(a[0]) # 97
print(a[1]) # 98
print(a[2]) # 99
```

When you write `b"abc"` in python, the three characters are converted into binary according to ASCII table. ASCII value
for letter "a" is 97, "b" is 98 and so on. We are storing three binary bytes. memoryview allows us to access
those bytes as if they were just a list of numbers. This "convenience" could be a source of confusion. The underlying
data is actually 3 bytes of data (`1100001 1100010 1100011`). But we are creating them by typing `b"abc"`, which is
implicitly converted to the binary form. Then when we are accessing them, the binary form is converted back, but
in the ordinals of ASCII table. (letter "a" is 1100001, which is 97 in decimal).

We can look at the length of this memoryview, which is the number of element (3). We can also check the size of each
element, which is 1 byte. We can also check the actual total size, which is 3 bytes.

```python
print(len(a)) # 3
print(a.itemsize) # 1 (byte)
print(a.nbytes) # 3 (byte)
```

Our `a` memoryview has a format. It specifies how to interpret the elements it holds. To see the current format:

```python
print(a.format) # "B"
```

"B" means "unsigned char" in C language. It refers to 8 bit integer that ranges from 0 to 255 and is a common choice
to hold ASCII characters. You can see what the format character means by referring to this
[table](https://docs.python.org/3/library/struct.html#format-characters).

Storing text is boring, a more common example is to get a hold of numpy array, and pass it to GPU.

```python
import numpy as np
a = np.array([1,2,3,4])
print(a.dtype) # np.int64
print(a.nbytes) # 32
print(a.itemsize) # 8
```

We have initialized a numpy buffer, the datatype is int64, meaning each number occupies 64 bits (or 8 bytes), and
the total size of this buffer is 32 bytes. We want to pass it to GPU, but if you look at the documentation for Metal
or CUDA, you'll notice that they take in a pointer:

```c++
- (id<MTLBuffer>)newBufferWithBytes:(const void *)pointer 
                             length:(NSUInteger)length 
                            options:(MTLResourceOptions)options;
```

That means we need to have a pointer to the numpy buffer. That pointer is the memoryview object. Numpy provides a way
to access it directly:

```python
ptr = a.data
print(ptr) # <memory at 0x109ca3d00>
print(ptr.format) # l (l is the format for int64, also known as long)
print(ptr.itemsize) # 8
print(ptr.nbytes) # 32

# You can even index into the data directly:
print(ptr[0]) # 1
```

The fact you can index into this memory segment as if it's a regular python list is both a convenience and a confusion.
It means that the memoryview object is not a plain pointer. It has additional metadata that tells downstream function
to interpret it as 4 elements 8 bytes each. So if you try to pass it to the GPU library, it will complain. As a result,
the memoryview need to be cast to a format that holds nothing more but a plain pointer to some bytes. And plain bytes
is also known as unsigned char:

```python
casted_ptr = ptr.cast("B")
print(casted_ptr.itemsize) # 1
print(len(casted_ptr)) # 32
print(casted_ptr.nbytes) # 32
print(casted_ptr[0]) # 1
print(casted_ptr[1]) # 0
print(casted_ptr[2]) # 0 
print(casted_ptr[3]) # 0
```

Now when you index into the buffer, the length becomes 32, each refers to a byte. We can grab the first four bytes, and
see that it has "1", "0", "0", "0" respectively. This refers to `0000 0001 0000 0000 0000 0000 0000 0000` in binary,
and is how number 1 is stored in int64 format. Note that because of endianess, the lowest byte is shown first.

Sometimes a GPU library may return the result in the format of a pointer as well. When that pointer arrives at the python
runtime, it will be a memoryview. Numpy allows us to construct an array using that memoryview, so we can convert our
pointer back.

Recall that we have two pointers now, the int64 pointer, and the casted binary pointer. Using the int64 pointer, we can
pass it directly to numpy, and it will see the int64 metadata and interpret the result correctly:

```python
print(np.array(ptr)) # [1 2 3 4]

# Using asarray will not create a separate copy of the underlying memory
print(np.asarray(ptr)) # [1 2 3 4]
```


If you pass the casted ptr, the result will look off:

```python
print(np.array(casted_ptr)) # [1 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0]
```

It's because "B" data type refers to a block of bytes, so intepreting as bytes result in 32 elements. This is where
we have to add additional hints so numpy knows how to interpret them:

```python
print(np.array(casted_ptr).view(np.int64)) # [1,2,3,4]
```

Or alternatively, cast the memoryview itself:

```python
print(np.array(casted_ptr.cast("l"))) # [1 2 3 4]
```
