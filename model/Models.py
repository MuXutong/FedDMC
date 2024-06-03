import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 784)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 5, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 5)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

class EMnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 5, 100)
        self.fc2 = nn.Linear(100, 62)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 5)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor



if __name__ == '__main__':
    x = torch.randn(32, 3, 32, 32)
    net = models.resnet18(num_classes=10)
    print(net)
    Y = net(x)
    print(Y.shape)
    # x = torch.randn(32, 1, 28, 28)
    # print(x.shape)
    # net = Mnist_CNN()
    #
    # print(net)
    # Y = net(x)
    # print(Y.shape)
    #
