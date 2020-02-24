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
        width = 50
        self.r1 = ResidLayer(width)
        self.r2 = ResidLayer(width)
        self.r3 = ResidLayer(width)
        self.r4 = ResidLayer(width)

    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        return x
