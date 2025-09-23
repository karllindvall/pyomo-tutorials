

import tempfile
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.onnx
from omlt.io.onnx import(
    write_onnx_model_with_bounds,
    load_onnx_neural_network_with_bounds,
)

from OMLT_workflow import bounds_from_array

def input_dim(model:torch.nn.Module):
    """Infer the input dimension of the model from the model structure if possible."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            return int(m.in_features)
    return None

# def torch_to_onnx(
#     models
# ):