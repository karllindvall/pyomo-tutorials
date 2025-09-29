'Trains and exports a PyTorch model to OMLT in ONNX format.'

import tempfile

import torch
import torch.onnx
from omlt.io.onnx import(
    write_onnx_model_with_bounds,
    load_onnx_neural_network_with_bounds,
)

from data import bounds_from_array, scale_dataset, load_csv
from models import SigmoidNN, ReluNN, MixedNN
from training import make_dataloader, train_model

def export_to_omlt(model, input_bounds, input_dim=1):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        x = torch.randn(10, input_dim)
        torch.onnx.export(
            model,
            x,
            f,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12,
        )
        write_onnx_model_with_bounds(f.name, onnx_model=None, input_bounds=input_bounds)
        net = load_onnx_neural_network_with_bounds(f.name)
    return net

df = scale_dataset(load_csv(r"C:\Users\karlj\OneDrive\Documents\GitHub\pyomo-tutorials\exercises\Custom\OMLT_tutorials\data\sin_quadratic.csv"))
X = df["x_scaled"].values
Y = df["y_scaled"].values
loader = make_dataloader(X, Y, batch_size=32)

nn1 = SigmoidNN()
nn2 = ReluNN()
nn3 = MixedNN()

train_model(nn1, loader, epochs=75)
train_model(nn2, loader, epochs=75)
train_model(nn3, loader, epochs=150)

input_bounds = bounds_from_array(X)

net_sigmoid = export_to_omlt(nn1, input_bounds, input_dim=1)
net_relu = export_to_omlt(nn2, input_bounds, input_dim=1)
net_mixed = export_to_omlt(nn3, input_bounds, input_dim=1)