import torch
from torchvision import datasets, transforms

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)
w2 = torch.randn(256, 10)
b2 = torch.randn(10)

for img in iter(trainloader):
    image = img[0].view(img[0].shape[0], -1)
    l1 = sigmoid( image @ w1 + b1 )
    l2 = sigmoid( l1 @ w2 + b2 )
    probabilities = softmax(l2)

    # Does it have the right shape? Should be (64, 10)
    print(probabilities.shape)
    # Does it sum to 1?
    print(probabilities.sum(dim=1))
