import torch
from torch import nn
from torch.nn import functional as F

class ResidLayer(nn.Module):
    def __init__(self, size):
        super(ResidLayer, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        r = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x + r
        x = F.relu(x)
        return x

class RubiksNetwork(nn.Module):
    def __init__(self):
        super(RubiksNetwork, self).__init__()
        width = 1000
        self.fc1 = nn.Linear(54, 5000)
        self.fc2 = nn.Linear(5000, width)
        self.r1 = ResidLayer(width)
        self.r2 = ResidLayer(width)
        self.r3 = ResidLayer(width)
        self.r4 = ResidLayer(width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.fc3(x)
        return x
