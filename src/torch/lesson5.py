from fastai.basics import *

path = Path('../../datasets')
print(path.ls())

with gzip.open(path / 'mnist.pkl.gz', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
for i in [0, 1,]:
    plt.imshow(x_train[i].reshape((28, 28)), cmap="Blues")
    plt.show()
    print(i)
print(x_train.shape)

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

bs = 64
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
data = DataBunch.create(train_ds, valid_ds, bs=bs)

x, y = next(iter(data.train_dl))
print(x.shape, y.shape)


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10, bias=True)

    def forward(self, x_batch): return self.lin(x_batch)


model = Mnist_Logistic()
print(model.lin)
print(model)

# print model parameters
for p in model.parameters():
    print(p.shape)

lr = 2e-2

loss_func = nn.CrossEntropyLoss()

def update(x,y,lr):
    wd = 1e-5
    y_hat = model(x)
    # weight decay
    w2 = 0.
    for p in model.parameters(): w2 += (p**2).sum()
    # add to regular loss
    loss = loss_func(y_hat, y) + w2*wd
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.sub_(lr * p.grad)
            p.grad.zero_()
    return loss.item()

losses = [update(x,y,lr) for x,y in data.train_dl]
plt.plot(losses);
plt.show()


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 50, bias=True)
        self.lin2 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        x = self.lin1(xb)
        x = F.relu(x)
        return self.lin2(x)

model = Mnist_NN()
losses = [update(x,y,lr) for x,y in data.train_dl]
plt.plot(losses);
plt.show()


model = Mnist_NN()
def update(x,y,lr):
    opt = optim.Adam(model.parameters(), lr)
    y_hat = model(x)
    loss = loss_func(y_hat, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()

losses = [update(x,y,1e-3) for x,y in data.train_dl]

plt.plot(losses);
plt.show()
