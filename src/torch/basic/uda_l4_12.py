from torch import torch, nn, optim
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.FashionMNIST('datasets/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

import matplotlib.pyplot as plt
img = next(iter(trainloader))[0]
plt.imshow(img[0][0], cmap="Blues")
plt.show()

neurons = 512

class NNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, neurons)
        self.hidden2 = nn.Linear(neurons, neurons)
        self.activation = nn.ReLU()
        self.output = nn.Linear(neurons, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout(self.activation(x))
        self.hidden2(x)
        x = self.dropout(self.activation(x))
        x = self.output(x)
        return self.softmax(x)


# or using Sequential
from collections import OrderedDict

# model1 = nn.Sequential(OrderedDict([
#     ('hidden', nn.Linear(784, 256)),
#     ('relu', nn.ReLU()),
#     ('output', nn.Linear(256, 10)),
#     ('softmax', nn.Softmax(dim=1))]))

# Defining the loss
criterion = nn.NLLLoss()

model = NNetwork()
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 2
for i in range(epochs):
    running_loss = 0
    for img, label in iter(trainloader):
        img = img.view(img.shape[0], -1)
        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

with torch.no_grad():
    test_img = next(iter(trainloader))
    result = model.forward(test_img[0].view(test_img[0].shape[0], -1))
    print(torch.argmax(result[0]), test_img[1][0])
