# Tutorials on Tinygrad 

[View on Github](https://github.com/mesozoic-egg/tinygrad-notes) |
[View on Website](https://mesozoic-egg.github.io/tinygrad-notes)

A series of tutorials/study-notes that help you understand the internals of tinygrad
and equip you to start contributing to it. 
The [quickstart guide](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md)
and [abstraction guide](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py)
are great resources, but may not be so beginner friendly, the following might be
more digestable.

Fundamentals (better read in orders):

1. [High level overview of the abstractions](dotproduct.md)

1. [How kernel fusion works - ScheduleItem](scheduleitem.md)

1. [The intermediate representation (IR)](uops.md)

1. [How GPU code is generated - backends and runtime](backends.md) 


Miscellaneous topics:

1. [Shapetracker allows for zero cost movement ops](shapetracker.md) 

1. [How dimension merging works](mergedim.md)

1. [How to profile a run and tune the performance](profiling.md)

1. [How JIT and cache enable faster compilation - TinyJit](jit.md)

1. [Command queue is a better way to handle job dispatch](commandqueue.md)

1. [Multi GPU training](multigpu.md)

1. [Adding custom accelerator](addingaccelerator.md)

1. [Code generator](codegen.md) and [details on common UOPS](uops-doc.md)

1. [Tensor core support part 1](cuda-tensor-core-pt1.md)

1. [Loop unrolling (upcast)](upcast.md) and [the underlying Symbolic library](upcast2.md)

1. [Interpreting colors and numbers in kernel names](colors.md)

Bounty explanations:

1. [Symbolic mean](symbolic-mean.md)

## What is tinygrad?

Tinygrad stands out as a deep learning framework, akin to Pytorch, XLA, and ArrayFire, 
yet it distinguishes itself by being more user-friendly, swifter, and less presumptive 
about the specifics of your hardware.

Mirroring Pytorch's user-friendly frontend, Tinygrad enhances model training and 
inference efficiency by employing lazy evaluation on the GPU. This approach compiles 
your model into highly optimized GPU code, capable of extending across multiple devices, 
thereby optimizing both time and financial resources.

Moreover, it offers a significant benefit by separating the machine learning software 
from the computing hardware. 
Many ML frameworks are designed primarily for CUDA, implying an expectation of
execution on Nvidia GPUs. This assumption can hinder the transition to alternative 
hardware in the future. Given the rapid advancements and competitive pricing strategies 
employed by numerous GPU manufacturers to offer comparable computing power at lower 
costs, ensuring your software stack is hardware-agnostic becomes an essential strategy 
for future-proofing.

This is where tinygrad truly shines. Our approach involves compiling machine 
learning models into a highly optimized Intermediate Representation (IR), which 
we then translate directly into GPU-specific instructions. Our goal is to drill 
down to the lowest possible level of instruction: PTX for Nvidia, KFD for AMD, 
and Metal for Apple devices. By targeting the foundational layers of the stack, 
we not only enhance compatibility across various hardware platforms but also unlock 
significant performance improvements. Additionally, this strategy leads to enhanced 
system stability and a reduction in the ongoing maintenance efforts.
