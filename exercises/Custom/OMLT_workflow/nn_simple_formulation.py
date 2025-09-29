import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements
import omlt
from omlt import OmltBlock
from omlt.neuralnet import (
    ReluBigMFormulation,
    ReluComplementarityFormulation,
)

from data import bounds_from_array, scale_dataset, load_csv
from training import make_dataloader, train_model
from onnx_constructor import export_to_omlt

from models import SimpleNN

# Train PyTorch model on imported data and export as ONNX

df = scale_dataset(load_csv(r"C:\Users\karlj\OneDrive\Documents\GitHub\pyomo-tutorials\exercises\Custom\OMLT_tutorials\data\sin_quadratic.csv"))
X = df["x_scaled"].values
Y = df["y_scaled"].values
loader = make_dataloader(X, Y, batch_size=32)

nn4 = SimpleNN()
train_model(nn4, loader, epochs=75)

input_bounds = bounds_from_array(X)
net_simple = export_to_omlt(nn4, input_bounds, input_dim=1)

# Extract weights and biases for algebraic formulation

W1 = nn4.d1.weight.detach().numpy().flatten()       # (3, 1) -> (3, )
b1 = nn4.d1.bias.detach().numpy()                   # (3, )
W2 = nn4.out.weight.detach().numpy().flatten()      # (3, 1) -> (3, )
b2 = nn4.out.bias.detach().numpy()                  # (1, )



# Define pyomo model with ReLU complimentary constraints

model4_comp = pyo.ConcreteModel()
model4_comp.x = pyo.Var(initialize=0)
model4_comp.y = pyo.Var(initialize=0)
model4_comp.obj = pyo.Objective(expr=(model4_comp.y))
model4_comp.nn = OmltBlock()

formulation4_comp = ReluComplementarityFormulation(net_simple)
model4_comp.nn.build_formulation(formulation4_comp)

@model4_comp.Constraint()
def connect_inputs(m):
    return m.x == m.nn.inputs[0]

@model4_comp.Constraint()
def connect_outputs(m):
    return m.y == m.nn.outputs[0]

status_4_comp = pyo.SolverFactory("ipopt").solve(model4_comp, tee=False)
solution_4_comp = (pyo.value(model4_comp.x), pyo.value(model4_comp.y))



# Define model with binary variables and BigM constraints

model4_bigm = pyo.ConcreteModel()
model4_bigm.x = pyo.Var(initialize=0)
model4_bigm.y = pyo.Var(initialize=0)
model4_bigm.obj = pyo.Objective(expr=(model4_bigm.y))
model4_bigm.nn = OmltBlock()

formulation4_bigm = ReluBigMFormulation(net_simple)
model4_bigm.nn.build_formulation(formulation4_bigm)

@model4_bigm.Constraint()
def connect_inputs(m):
    return m.x == m.nn.inputs[0]

@model4_bigm.Constraint()
def connect_outputs(m):
    return m.y == m.nn.outputs[0]

status_4_bigm = pyo.SolverFactory("cbc").solve(model4_bigm, tee=False)
solution_4_bigm = (pyo.value(model4_bigm.x), pyo.value(model4_bigm.y))



# Defining algebraic implementation of Complimentarity formulation

x_bounds = input_bounds[0]

modela_comp = pyo.ConcreteModel()
modela_comp.x = pyo.Var(bounds=x_bounds)        # setting bounds
modela_comp.z = pyo.Var(range(3))
modela_comp.h = pyo.Var(range(3))
modela_comp.y = pyo.Var()
modela_comp.obj = pyo.Objective(expr=(modela_comp.y))

# Hidden layer
def z_def(m, j):
    return m.z[j] == W1[j]*m.x + b1[j]                          # algebraic formulation of layer
modela_comp.z_def_con = pyo.Constraint(range(3), rule=z_def)    # range(3) for three hidden nodes

# Output layer
def y_def(m):
    return m.y == sum(W2[j]*m.h[j] for j in range(3)) + b2      # algebraic formulation for output node
modela_comp.y_def_con = pyo.Constraint(rule=y_def)

# ReLU Complimentary formulation
def relu_def(m, j):
    return complements(0 <= m.h[j], m.h[j] - m.z[j] >= 0)       # complimentarity conditions
modela_comp.relu_con = Complementarity(range(3), rule=relu_def)

status_a_comp = pyo.SolverFactory("ipopt").solve(modela_comp, tee=False)
solution_a_comp = (pyo.value(modela_comp.x), pyo.value(modela_comp.y))



# Define algebraic implementation using BigM formulation

# Bounds for BigM
def z_bounds(w, b, xL, xU):
    low = min(w*xL + b, w*xU + b)
    high = max(w*xL + b, w*xU + b)
    return low, high

zL, zU = zip(*[z_bounds(W1[j], b1[j], *x_bounds) for j in range(3)])

modela_bigm = pyo.ConcreteModel()
modela_bigm.x = pyo.Var(bounds=x_bounds)
modela_bigm.z = pyo.Var(range(3))
modela_bigm.h = pyo.Var(range(3))
modela_bigm.y = pyo.Var()
modela_bigm.delta = pyo.Var(range(3), within=pyo.Binary) # binary variable for BigM constraints
modela_bigm.obj = pyo.Objective(expr=(modela_bigm.y))

# Hidden layer
def z_def_bigm(m, j):
    return m.z[j] == W1[j]*m.x + b1[j]
modela_bigm.z_def_con = pyo.Constraint(range(3), rule=z_def_bigm)

# Output layer
def y_def_bigm(m):
    return m.y == sum(W2[j]*m.h[j] for j in range(3)) + b2
modela_bigm.y_def_con = pyo.Constraint(range(3), rule=y_def_bigm)

# BigM constraints
@modela_bigm.Constraint()
def relu1(m, j):
    return m.h[j] >= 0

@modela_bigm.Constraint()
def relu2(m, j):
    return m.h[j] >= m.z[j]

@modela_bigm.Constraint()
def relu3(m, j):
    return m.h[j] <= m.z[j] - (1 - m.delta[j])*zL[j]

@modela_bigm.Constraint()
def relu4(m, j):
    return m.h[j] <= m.delta[j]*zU[j]

status_a_bigm = pyo.SolverFactory("cbc").solve(modela_bigm, tee=False)
solution_a_bigm = (pyo.value(modela_bigm.x), pyo.value(modela_bigm.y))


print("ReLU Complementarity Solution:")
print("# of variables: ", model4_comp.nvariables())
print("# of constraints: ", model4_comp.nconstraints())
print("x = ", solution_4_comp[0])
print("y = ", solution_4_comp[1])
print("Solve Time: ", status_4_comp["Solver"][0]["Time"])

print("ReLU BigM Solution:")
print("# of variables: ", model4_bigm.nvariables())
print("# of constraints: ", model4_bigm.nconstraints())
print("x = ", solution_4_bigm[0])
print("y = ", solution_4_bigm[1])

print("ReLU Complementarity Solution:")
print("# of variables: ", modela_comp.nvariables())
print("# of constraints: ", modela_comp.nconstraints())
print("x =", solution_a_comp[0])
print("y =", solution_a_comp[1])

print("ReLU Big-M Solution:")
print("# of variables: ", modela_bigm.nvariables())
print("# of constraints: ", modela_bigm.nconstraints())
print("x =", solution_a_bigm[0])
print("y =", solution_a_bigm[1])

