from torch import torch, nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the test data
testset = datasets.FashionMNIST('datasets/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class NNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(256, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        x = self.dropout(self.sigmoid(x))
        x = self.output(x)
        return self.softmax(x)


state_dict = torch.load('models/uda_l4_14.pth')
model = NNetwork()
model.load_state_dict(state_dict)

img_batch = next(iter(testloader))
result = model.forward(img_batch[0].view(img_batch[0].shape[0], -1))
print(torch.argmax(result, dim=1)==img_batch[1])
