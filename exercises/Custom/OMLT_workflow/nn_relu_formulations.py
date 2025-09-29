import pyomo.environ as pyo
import omlt
from omlt import OmltBlock
from omlt.neuralnet import (
    ReluBigMFormulation,
    ReluComplementarityFormulation,
)

from data import bounds_from_array, scale_dataset, load_csv
from training import make_dataloader, train_model
from onnx_constructor import export_to_omlt

from models import ReluNN

# Train PyTorch model on imported data and export as ONNX

df = scale_dataset(load_csv(r"C:\Users\karlj\OneDrive\Documents\GitHub\pyomo-tutorials\exercises\Custom\OMLT_tutorials\data\sin_quadratic.csv"))
X = df["x_scaled"].values
Y = df["y_scaled"].values
loader = make_dataloader(X, Y, batch_size=32)

nn2 = ReluNN()
train_model(nn2, loader, epochs=75)

input_bounds = bounds_from_array(X)
net_relu = export_to_omlt(nn2, input_bounds, input_dim=1)

# Define pyomo model with ReLU complimentary constraints

model2_comp = pyo.ConcreteModel()
model2_comp.x = pyo.Var(initialize=0)
model2_comp.y = pyo.Var(initialize=0)
model2_comp.obj = pyo.Objective(expr=(model2_comp.y))
model2_comp.nn = OmltBlock()

formulation2_comp = ReluComplementarityFormulation(net_relu)
model2_comp.nn.build_formulation(formulation2_comp)

@model2_comp.Constraint()
def connect_inputs(m):
    return m.x == m.nn.inputs[0]

@model2_comp.Constraint()
def connect_outputs(m):
    return m.y == m.nn.outputs[0]

status_2_comp = pyo.SolverFactory("ipopt").solve(model2_comp, tee=False)
solution_2_comp = (pyo.value(model2_comp.x), pyo.value(model2_comp.y))

# Define model with binary variables and BigM constraints

model2_bigm = pyo.ConcreteModel()
model2_bigm.x = pyo.Var(initialize=0)
model2_bigm.y = pyo.Var(initialize=0)
model2_bigm.obj = pyo.Objective(expr=(model2_bigm.y))
model2_bigm.nn = OmltBlock()

formulation2_bigm = ReluBigMFormulation(net_relu)
model2_bigm.nn.build_formulation(formulation2_bigm)

@model2_bigm.Constraint()
def connect_inputs(m):
    return m.x == m.nn.inputs[0]

@model2_bigm.Constraint()
def connect_outputs(m):
    return m.y == m.nn.outputs[0]

status_2_bigm = pyo.SolverFactory("cbc").solve(model2_bigm, tee=False)
solution_2_bigm = (pyo.value(model2_bigm.x), pyo.value(model2_bigm.y))


print("ReLU Complementarity Solution:")
print("# of variables: ", model2_comp.nvariables())
print("# of constraints: ", model2_comp.nconstraints())
print("x = ", solution_2_comp[0])
print("y = ", solution_2_comp[1])
print("Solve Time: ", status_2_comp["Solver"][0]["Time"])

print("ReLU BigM Solution:")
print("# of variables: ", model2_bigm.nvariables())
print("# of constraints: ", model2_bigm.nconstraints())
print("x = ", solution_2_bigm[0])
print("y = ", solution_2_bigm[1])
print("Solve Time: ", status_2_bigm["Solver"][0]["Time"])