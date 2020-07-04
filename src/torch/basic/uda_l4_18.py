from torch import torch, nn, optim
from torchvision import datasets, transforms
from src.torch.torch_models.fc_model import NNetwork
import matplotlib.pyplot as plt

# Define a transform to normalize the data
train_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      # transforms.RandomRotation(30),
                                      # transforms.RandomResizedCrop(224),
                                      # transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                      ])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

# Download and load the test data
trainset = datasets.ImageFolder('datasets/Cat_Dog_data/train', transform=train_transform)
testset = datasets.ImageFolder('datasets/Cat_Dog_data/test', transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# Defining the loss
criterion = nn.NLLLoss()

model = NNetwork(150528)
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 2
train_losses, test_losses = [], []

for i in range(epochs):
    running_loss = 0
    for image, label in trainloader:
        output = model(image)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                test_output = model(images)
                test_loss += criterion(test_output, labels)

                ps = torch.exp(test_output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()
    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))
    print("Epoch: {}/{}.. ".format(i + 1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
          "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
