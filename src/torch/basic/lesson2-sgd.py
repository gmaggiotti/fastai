from fastai.basics import *
import matplotlib.pyplot as plt

n = 100
x = torch.ones(n,2)
x[:,0].uniform_(-1.,1)
a = tensor([3.,2])
y = x**2@a + torch.rand(n)

def rmse(y_hat, y): return ((y_hat - y) ** 2).mean()
a = nn.Parameter(a)

def update():
    y_hat = x**2@a
    loss = rmse(y, y_hat)
    if t % 10 == 0: print(loss)
    loss.backward()
    with torch.no_grad():
        # Calculate de grad of loss using lr and substract them from a
        a.sub_(lr * a.grad)
        a.grad.zero_()

lr = 1e-1
for t in range(1000): update()

plt.scatter(x[:,0],y)
plt.scatter(x[:,0],(x**2@a).detach().numpy());
plt.show()