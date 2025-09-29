import pyomo.environ as pyo
import omlt
from omlt import OmltBlock
from omlt.neuralnet import (
    FullSpaceNNFormulation,  
)
from omlt.neuralnet.activations import ComplementarityReLUActivation

from data import bounds_from_array, scale_dataset, load_csv
from training import make_dataloader, train_model
from onnx_constructor import export_to_omlt

from models import MixedNN

# Train PyTorch model on imported data and export as ONNX

df = scale_dataset(load_csv(r"C:\Users\karlj\OneDrive\Documents\GitHub\pyomo-tutorials\exercises\Custom\OMLT_tutorials\data\sin_quadratic.csv"))
X = df["x_scaled"].values
Y = df["y_scaled"].values
loader = make_dataloader(X, Y, batch_size=32)

nn3 = MixedNN()
train_model(nn3, loader, epochs=75)

input_bounds = bounds_from_array(X)
net_mixed = export_to_omlt(nn3, input_bounds, input_dim=1)

# Define model with full-space and complementary formulations

model3_mixed = pyo.ConcreteModel()
model3_mixed.x = pyo.Var(initialize=0)
model3_mixed.y = pyo.Var(initialize=0)
model3_mixed.obj = pyo.Objective(expr=(model3_mixed.y))
model3_mixed.nn = OmltBlock()

# Override standard BigM formulation and use complimentary formulation
formulation3_mixed = FullSpaceNNFormulation(
    net_mixed, activation_constraints={"relu": ComplementarityReLUActivation()}
)
model3_mixed.nn.build_formulation(formulation3_mixed)

@model3_mixed.Constraint()
def connect_inputs(m):
    return m.x == m.nn.inputs[0]

@model3_mixed.Constraint()
def connect_outputs(m):
    return m.y == m.nn.outputs[0]

status_3_mixed = pyo.SolverFactory("ipopt").solve(model3_mixed, tee=False)
solution_3_mixed = (pyo.value(model3_mixed.x), pyo.value(model3_mixed.y))

print("Mixed NN Solution:")
print("# of variables: ", model3_mixed.nvariables())
print("# of constraints: ", model3_mixed.nconstraints())
print("x = ", solution_3_mixed[0])
print("y = ", solution_3_mixed[1])
print("Solve Time: ", status_3_mixed["Solver"][0]["Time"])