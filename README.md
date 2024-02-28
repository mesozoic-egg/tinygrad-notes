# From micrograd to tinygrad

This is a series of google colab notebook that helps you understand how
tinygrad is built and hopefully make it easier for you to start
contributing to that project.

The starting ground is micrograd, a simple autograd library that handles
scalar values. You can find a 2 hour tutorial online on how it's built
and how it works. 

We will start from an even simplier version of it, and show how you can warp
it into a full featured autograd libary that can JIT compile, run on GPU,
and even more.