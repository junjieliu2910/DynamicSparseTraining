import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .binarization import *


class LeNet_300_100_Masked(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = MaskedMLP(784, 300)
        self.fc2 = MaskedMLP(300, 100)
        self.fc3 = MaskedMLP(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNet_5_Masked(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = MaskedConv2d(1, 20, (5, 5), 1)
        self.conv2 = MaskedConv2d(20, 50, (5, 5), 1)
        self.fc3 = MaskedMLP(4 * 4 * 50, 500)
        self.fc4 = MaskedMLP(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.fc3(x.view(-1, 4 * 4 * 50)))
        
        return self.fc4(x)


class LeNet_300_100(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, 5 , 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc3 = nn.Linear(4 * 4 * 50, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.fc3(x.view(-1, 4 * 4 * 50)))
        
        return self.fc4(x)