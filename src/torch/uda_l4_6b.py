import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import numpy as np

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


class NNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(256, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return self.softmax(x)


# or using Sequential
from collections import OrderedDict

model1 = nn.Sequential(OrderedDict([
    ('hidden', nn.Linear(784, 256)),
    ('sigmoid', nn.Sigmoid()),
    ('output', nn.Linear(256, 10)),
    ('softmax', nn.Softmax(dim=1))]))

# Defining the loss
criterion = nn.CrossEntropyLoss()

model = NNetwork()
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 2
for i in range(epochs):
    running_loss = 0
    for img, label in iter(trainloader):
        image = img.view(img.shape[0], -1)
        output = model.forward(image)
        # Calculate loss
        loss = criterion(output, label)

        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()
        # backward pass to calculate the new gradients
        loss.backward()
        # update the weights
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

with torch.no_grad():
    test_img = next(iter(trainloader))
    result = model.forward(test_img[0].view(test_img[0].shape[0], -1))
    print(torch.argmax(result[0]), test_img[1][0])
