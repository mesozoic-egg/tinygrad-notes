# Tutorials on Tinygrad

A series of tutorials that help you understand the internals of tinygrad
and be better equipped if you want to contribute to it.

The [quickstart guide](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md)
and [abstraction guide](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py)
already offer some really good resources on how to get started and the internals, 
but if you find the learning curve still too steep (it was to me at least). The 
following bite sized writeup might be a more friendly ramp. 

The following are better read in order:

1. [high level overview using dot product as example](dotproduct.md)

2. [how kernel fusion starts](scheduleitem.md)

3. [explaining the tinygrad IR](uops.md)

4. [tinygrad backend](backends.md) and [command queue](commandqueue.md)


Miscellaneous topics:
1. [shapetracker](shapetracker.md) and [dimension merging](mergedim.md)