import torch
import matplotlib.pyplot as plt
import numpy as np

x_ = np.random.uniform(-10,10, size=(100,))
x = torch.tensor(x_, requires_grad=True)
print(x)
y = x + 2
print(y)

z = y * y
out = z.mean()
print(z, out)

print("grad before:", x.grad)
out.backward()
print("grand after:", x.grad)

x1 = x.detach()
z1 = z.detach()
x_grad = x.grad.detach()

plt.scatter(x1, z1)
plt.scatter(x1, x_grad)
plt.show()
