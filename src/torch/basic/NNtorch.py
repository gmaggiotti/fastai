import torch
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + torch.exp(-x))


# input data, each column represent a dif neuron
ds = np.loadtxt("datasets/tinyds.csv", delimiter=",")
ds = torch.tensor(ds, dtype=torch.float)
X = ds[:, :-1]
y = ds[:, -1:]
epochs = 600
errors = []
max = X.view(X.shape).max()
X = 2 * X / float(max) - 1

torch.manual_seed(7)
w0 = 2 * torch.rand(X.shape[1], 1, dtype=torch.float) - 1  # mxn matrix of weights

# This is the main training loop. The output shows the evolution of the error between the model and desired. The
# error steadily decreases.
for j in range(epochs):

    # Calculate forward through the network.
    l1 = sigmoid(X @ w0)

    # Error back propagation of errors using the chain rule.
    l1_error = y - l1
    if (j % 10) == 0:  # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("epoch {}: {}".format(j, l1_error.mean().abs()))
    errors.append(l1_error.mean().abs().item())
    adjustment = l1_error * sigmoid(l1, deriv=True)  # (y-a).d/dw(-a), a = sigmoid(Sum Xi*Wi)

    # update weights (no learning rate term)
    w0 += X.T @ adjustment


def predict(X1):
    max = X1.view(X1.shape).max()
    l0 = 2 * X1 / float(max) - 1
    l1 = sigmoid(l0.float() @ w0)
    return l1.numpy()[0]


test_dataset = torch.tensor([1, 9, 19, 33, 16, 2, 1])
result = predict(test_dataset)
print("expected output 1, predicted output: " + repr(result))
assert (result > 0.95), "Test Failed. Expected result > 0.95"

test_dataset = torch.tensor([1, 0, 1, 4, 1, 3, 1])
result = predict(test_dataset)
print("expected output 0, predicted output " + repr(result))
assert (result < 0.95), "Test Failed. Expected result < 0.95"

xs = np.arange(0,epochs)
plt.scatter(xs,errors)
plt.show()