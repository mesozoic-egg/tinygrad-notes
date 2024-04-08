# Tutorials on Tinygrad 

[View on Github](https://github.com/mesozoic-egg/tinygrad-notes) |
[View on Website](https://mesozoic-egg.github.io/tinygrad-notes)

A series of tutorials that help you understand the internals of tinygrad
and be better equipped if you want to contribute to it. This is not an official tutorial, just my personal study notes, for now.

The [quickstart guide](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md)
and [abstraction guide](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py)
already offer some really good resources on how to get started and the internals, 
but if you find the learning curve still too steep (it was to me at least). The 
following bite sized writeup might be a more friendly ramp. 

The following are better read in order:

1. [High level overview of the abstractions](dotproduct.md)

2. [How kernel fusion works - ScheduleItem](scheduleitem.md)

3. [The intermediate representation (IR)](uops.md)

4. [How GPU code is generated - backends and runtime](backends.md) 


Miscellaneous topics to read after you finish the above, no particular order
is assumed:

1. [Shapetracker allows for zero cost movement ops](shapetracker.md) 

2. [How dimension merging works](mergedim.md)

2. [How to profile a run and tune the performance](profiling.md)

3. [How JIT and cache enable faster compilation - TinyJit](jit.md)

4. [Command queue is a better way to handle job dispatch](commandqueue.md)

5. [Multi GPU training](multigpu.md)

## What is tinygrad?

Tinygrad is a deep learning framework similar to Pytorch, XLA, arrayfire but
easier to use, faster, and make fewer assumptions about your hardware.

With a similar frontend to Pytorch that makes it very easy to use, 
Tinygrad also offers performant model training and running by evaluating everything
lazily, on the GPU. Your model is compiled into highly optimized GPU code that
can utilize multiple devices to squeeze out every flops out of your budget.

It also presents a clear advantage by decoupling the ML software and the compute. 
Most ML framework are built with CUDA in mind, they assume you will run on Nvidia 
GPU if you want the best performance. This makes it difficult to switch to other
hardware in the future. Considering that lots of GPU companies are playing catch
up aggressively, and also offering steep cash discount for a similar level of FLOPS,
it might be a good idea to future proof your software stack by making them vendor neutral.

That's where tinygrad comes to play, we compile machine learning model to a
highly optimized IR, and from there, we compile
directly to GPU instructions. We attempt to make the instruction as low level
as possible: on Nvidia that is PTX, on AMD that's KFD, on Apple that's Metal. 
The lower of the stack we can go, the easier it is to support different hardware
and allow for more performance boost. And when hardware companies fail to write
good userspace software, we will also gain better stability and reduce the
maintenance burden.