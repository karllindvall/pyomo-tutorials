'Importing a NN rather than building it from scratch using OMLT and the ONNX format.'

import pandas as pd

df = pd.read_csv(r"C:\Users\karlj\OneDrive\Documents\GitHub\pyomo-tutorials\exercises\Custom\OMLT\data\diabetes.csv")

# Split data into input and output
X = df.iloc[:, :8].to_numpy()
Y = df.iloc[:, 8].to_numpy()

# Setting bounds on variables for a tighter MIP formulation
import numpy as np

lb = np.min(X, axis=0)
ub = np.max(X, axis=0)
input_bounds = list(zip(lb, ub))

# Import fxn to write ONNX model and its bounds
from omlt.io import load_onnx_neural_network_with_bounds, write_onnx_model_with_bounds

# Building a PyTorch model
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_0 = nn.Linear(8, 12)
        self.dense_1 = nn.Linear(12, 1)
        self.out = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.dense_0(x))
        x = F.relu(self.dense_1(x))
        return self.out(x)
    
model = PyTorchModel()
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

dataset = TensorDataset(
    torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
)
dataloader = DataLoader(dataset, batch_size=10)

for epoch in range(150):
    for x_batch, y_batch in dataloader:
        y_batch_pred = model(x_batch)
        loss = loss_function(y_batch_pred, y_batch.view(*y_batch_pred.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch number: {epoch} loss: {loss.item()}")

# Exporting PyTorch model to ONNX
import torch.onnx
import tempfile

# Model input used for exporting
x = torch.randn(10, 8, requires_grad=True)
pytorch_model = None
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    torch.onnx.export(
        model,
        x,
        f,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    write_onnx_model_with_bounds(f.name, None, input_bounds)
    print(f"Wrote PyTorch model to {f.name}")
    pytorch_model = f.name

# Importing ONNX into OMLT
network_definition = load_onnx_neural_network_with_bounds(pytorch_model)

# This includes the bounds defined earlier (input_bounds)
network_definition.scaled_input_bounds
# Can iterate over layers to print input and output shape with activation function
for layer_id, layer in enumerate(network_definition.layers):
    print(f"{layer_id}\t{layer}\t{layer.activation}")
