from torch import nn

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