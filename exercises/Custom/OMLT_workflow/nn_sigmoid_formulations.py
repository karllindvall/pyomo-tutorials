import pyomo.environ as pyo
import omlt
from omlt import OmltBlock
from omlt.neuralnet import (
    FullSpaceSmoothNNFormulation,
    ReducedSpaceSmoothNNFormulation, 
)

from data import bounds_from_array, scale_dataset, load_csv
from training import make_dataloader, train_model
from onnx_constructor import export_to_omlt

from models import SigmoidNN

# Train PyTorch model on imported data and export as ONNX

df = scale_dataset(load_csv(r"C:\Users\karlj\OneDrive\Documents\GitHub\pyomo-tutorials\exercises\Custom\OMLT_tutorials\data\sin_quadratic.csv"))
X = df["x_scaled"].values
Y = df["y_scaled"].values
loader = make_dataloader(X, Y, batch_size=32)

nn1 = SigmoidNN()
train_model(nn1, loader, epochs=75)

input_bounds = bounds_from_array(X)
net_sigmoid = export_to_omlt(nn1, input_bounds, input_dim=1)

# Define reduced-space pyomo model

model1_reduced = pyo.ConcreteModel()
model1_reduced.x = pyo.Var(initialize=0)
model1_reduced.y = pyo.Var(initialize=0)

model1_reduced.obj = pyo.Objective(expr=(model1_reduced.y))

model1_reduced.nn = OmltBlock()

# Using reduced-space formulation
formulation1_reduced = ReducedSpaceSmoothNNFormulation(net_sigmoid)
model1_reduced.nn.build_formulation(formulation1_reduced)

# Connecting pyomo vars to NN
@model1_reduced.Constraint()
def connect_inputs(m):
    return m.x == m.nn.inputs[0]

@model1_reduced.Constraint()
def connect_outputs(m):
    return m.y == m.nn.outputs[0]

# Solve model
status_1_reduced = pyo.SolverFactory("ipopt").solve(model1_reduced, tee=False)
solution_1_reduced = (pyo.value(model1_reduced.x), pyo.value(model1_reduced.y))

# Define full-space pyomo model

model1_full = pyo.ConcreteModel()
model1_full.x = pyo.Var(initialize=0)
model1_full.y = pyo.Var(initialize=0)

model1_full.obj = pyo.Objective(expr=(model1_full.y))
model1_full.nn = OmltBlock()

formulation1_full = FullSpaceSmoothNNFormulation(net_sigmoid)
model1_full.nn.build_formulation(formulation1_full)

@model1_full.Constraint()
def connect_inputs(m):
    return m.x == m.nn.inputs[0]

@model1_full.Constraint()
def connect_outputs(m):
    return m.y == m.nn.outputs[0]

status_1_full = pyo.SolverFactory("ipopt").solve(model1_full, tee=False)
solution_1_full = (pyo.value(model1_full.x), pyo.value(model1_full.y))

print("Reduced Space Solution:")
print("# of variables: ", model1_reduced.nvariables())
print("# of constraints: ", model1_reduced.nconstraints())
print("x = ", solution_1_reduced[0])
print("y = ", solution_1_reduced[1])
print("Solve Time: ", status_1_reduced["Solver"][0]["Time"])

print("Full Space Solution:")
print("# of variables: ", model1_full.nvariables())
print("# of constraints: ", model1_full.nconstraints())
print("x = ", solution_1_full[0])
print("y = ", solution_1_full[1])
print("Solve Time: ", status_1_full["Solver"][0]["Time"])