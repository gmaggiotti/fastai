from __future__ import print_function
import torch
import numpy as np

# Construct a 5x3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix
rand_x = torch.rand(2, 2, dtype=torch.float)
print(rand_x)

# Construct a matrix filled zeros and of dtype long
zero_x = torch.zeros(2, 2, dtype=torch.long)
print(zero_x)

# get input data from a file
ds = np.loadtxt("../../datasets/tinyds.csv", delimiter=",")
ds_tensor = torch.tensor(ds, dtype=float)
print("a")

# torch operators
a = ds_tensor[1, :]
b = ds_tensor[2, :]
print("a:", a)
print("b", b)
print("a+b", a + b)

# resizing
c = torch.rand(4, 4)
c1 = c.view(16)
c2 = c.view(8, 2)
print(c, c1, c2)

# get a value form a tensor
c.numpy()[0, 0]
c[0, 0].item()

# converting from tensor to numpy it shares same memory location
c_numpy = c[0, :].numpy()
c.add_(1)
print("converting tensorc[0: to numpy")
print(c_numpy, c[0, :])

# convert form numpy to torch tensor
d_numpy = np.array([1,2])
d = torch.from_numpy(d_numpy)

# matrix and tensor mul
m1 = torch.tensor([[1,1],[2,2]])
m2 = torch.tensor([[1],[0]])

print(m1@m2)

# map a list integers of into a list of tensors
arr = [1,2,3]
a = [tensor for tensor in map(torch.tensor, arr)]
b = list(map(torch.tensor,arr))
assert a == b