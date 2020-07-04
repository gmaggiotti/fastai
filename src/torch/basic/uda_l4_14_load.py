from torch import torch
from torchvision import datasets, transforms
from src.torch.torch_models.fc_model import NNetwork

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the test data
testset = datasets.FashionMNIST('datasets/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

state_dict = torch.load('models/uda_l4_14.pth')
model = NNetwork()
model.load_state_dict(state_dict)

img_batch = next(iter(testloader))
result = model.forward(img_batch[0].view(img_batch[0].shape[0], -1))
print(torch.argmax(result, dim=1) == img_batch[1])
