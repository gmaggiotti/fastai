from torch import torch, nn, optim
from torchvision import datasets, transforms, models
from src.torch.torch_models.fc_model import NNetwork
import matplotlib.pyplot as plt


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

# Define a transform to normalize the data
train_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                             ])

# initilize a pretrained sensenet121 model
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
freeze_model(model)

# add a classifier model as last layers
# take in consideratio the output len of the sensenet121 which is 1024
model.classifier = NNetwork(1024)

# Download and load the test data
trainset = datasets.ImageFolder('datasets/Cat_Dog_data/train', transform=train_transform)
testset = datasets.ImageFolder('datasets/Cat_Dog_data/test', transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# Defining the loss
criterion = nn.NLLLoss()

# Optimizers require only the parameters of the classifier to optimize and a learning rate
optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)

epochs = 5
train_losses, test_losses = [], []

for i in range(epochs):
    running_loss = 0
    for ii, (image, label) in enumerate(trainloader):
        output = model(image)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if ii==3:
            break

    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for ii, (image, label) in enumerate(testloader):
            test_output = model(image)
            test_loss += criterion(test_output, label)

            ps = torch.exp(test_output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == label.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            if ii==3:
                break
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
