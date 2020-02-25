import torch
from torch import nn
from torch.nn import functional as F

class ResidLayer(nn.Module):
    def __init__(self, size):
        super(ResidLayer, self).__init__()
        self.fc = nn.Linear(size, size)

    def forward(self, x):
        return F.relu(self.fc(x) + x)

class RubiksNetwork(nn.Module):
    def __init__(self):
        super(RubiksNetwork, self).__init__()
        width = 200
        self.fc1 = nn.Linear(54, width)
        self.r1 = ResidLayer(width)
        self.r2 = ResidLayer(width)
        self.r3 = ResidLayer(width)
        self.r4 = ResidLayer(width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.fc2(x)
        return x
