from torch import Tensor
a = Tensor([2]).requires_grad_(True)
b = Tensor([3]).requires_grad_(True)
c = a * b
d = c * b
d.backward()
print(f"{a.grad=}")
print(f"{b.grad=}")