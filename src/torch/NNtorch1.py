from fastai.basics import *

path = Path('../../datasets')
print(path.ls())

X = np.loadtxt(path / 'train_dataset.csv', delimiter=',')
Y = np.loadtxt(path / 'label_dataset.csv', delimiter=',')

x_train = torch.tensor(X[:28], dtype=torch.float)
x_valid = torch.tensor(X[28:32], dtype=torch.float)
y_train = torch.tensor(Y[:28], dtype=torch.long)
y_valid = torch.tensor(Y[28:32], dtype=torch.long)

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
data = DataBunch.create(train_ds, valid_ds)


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(39, 39, bias=True)
        self.lin2 = nn.Linear(39, 1, bias=True)

    def forward(self, xb):
        x = self.lin1(xb)
        x = F.relu(x)
        return self.lin2(x)


lr = 2e-2
loss_func = nn.CrossEntropyLoss()


def update(x, y, lr):
    opt = optim.Adam(model.parameters(), lr)
    y_hat = model(x)
    loss = loss_func(y_hat, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


model = Mnist_NN()

losses = [update(x, y, lr) for x, y in data.train_dl]
plt.plot(losses);
plt.show()
