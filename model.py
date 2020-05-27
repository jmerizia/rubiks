import torch
from torch import nn
from torch.nn import functional as F

class ResidLayer(nn.Module):
    def __init__(self, size):
        super(ResidLayer, self).__init__()
        self.fc1 = nn.Linear(size, size)
        #self.bn1 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        #self.bn2 = nn.BatchNorm1d(size)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        #out = self.bn1(out)
        out = F.relu(out)

        out = self.fc2(out)
        #out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out

class RubiksNetwork(nn.Module):
    def __init__(self):
        super(RubiksNetwork, self).__init__()
        width = 100
        self.fc1 = nn.Linear(54, 100)
        #self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, width)
        #self.bn2 = nn.BatchNorm1d(width)
        self.r1 = ResidLayer(width)
        #self.r2 = ResidLayer(width)
        #self.r3 = ResidLayer(width)
        #self.r4 = ResidLayer(width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)

        x = self.r1(x)
        #x = self.r2(x)
        #x = self.r3(x)
        #x = self.r4(x)

        x = self.fc3(x)
        return x
