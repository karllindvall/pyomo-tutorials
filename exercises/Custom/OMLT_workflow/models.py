'Create three PyTorch models with different combinations of sigmoid and ReLU activation functions.'

import torch
from torch import nn
import torch.nn.functional as F

# Sigmoid NN
class SigmoidNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Linear(1, 50)
        self.d2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)
    def forward(self, x):
        x = torch.sigmoid(self.d1(x))
        x = torch.sigmoid(self.d2(x))
        return self.out(x)

# ReLU NN
class ReluNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Linear(1, 30)
        self.d2 = nn.Linear(30, 30)
        self.out = nn.Linear(30, 1)
    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return self.out(x)

# Mixed NN
class MixedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Linear(1, 50)
        self.d2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)
    def forward(self, x):
        x = torch.sigmoid(self.d1(x))
        x = F.relu(self.d2(x))
        return self.out(x)