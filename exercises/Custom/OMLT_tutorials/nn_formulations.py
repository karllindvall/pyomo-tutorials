'Different optimization formulations of NNs in Pyomo using OMLT'

# Data manipulation and plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Pyomo for optim and Pytorch for models
import pyomo.environ as pyo
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# OMLT for interface between NN and pyomo
import omlt
from omlt import OmltBlock
import torch.onnx
import tempfile
import onnx
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from omlt.neuralnet import (
    FullSpaceNNFormulation,
    FullSpaceSmoothNNFormulation,
    ReducedSpaceSmoothNNFormulation,
    ReluBigMFormulation,
    ReluComplementarityFormulation,
    ReluPartitionFormulation,
)
from omlt.neuralnet.activations import ComplementarityReLUActivation

df = pd.read_csv(r"C:\Users\karlj\OneDrive\Documents\GitHub\pyomo-tutorials\exercises\Custom\OMLT\data\sin_quadratic.csv", index_col=[0]);

x = df["x"]
y = df["y"]

# Mean, std dev., and scaled data
mean_data = df.mean(axis=0)
std_data = df.std(axis=0)
df["x_scaled"] = (df["x"] - mean_data["x"]) / std_data["x"]
df["y_scaled"] = (df["y"] - mean_data["y"]) / std_data["y"]

# Creating neural network formulations

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
    
# Instantiating networks
nn1 = SigmoidNN()
nn2 = ReluNN()
nn3 = MixedNN()

# Loading data and training networks

# Converting to torch tensors
X = torch.tensor(df["x_scaled"].values, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(df["y_scaled"].values, dtype=torch.float32).view(-1, 1)

# Creating dataset and loader
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
def train_model(model, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        epoch_loss = running_loss / len(dataset)
        history.append(epoch_loss)

    return history

# Train all three models
history1 = train_model(nn1, epochs=75)
history2 = train_model(nn2, epochs=75)
history3 = train_model(nn3, epochs=150)
