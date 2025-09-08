import pyomo.environ as pyo

## Initialize model ##
num_points = 10
model = pyo.ConcreteModel()
model.points = pyo.RangeSet(0, num_points) # Discretization points
model.h = pyo.Param(initialize=1.0/num_points)

model.z = pyo.Var(model.points)
model.dzdt = pyo.Var(model.points)

model.obj = pyo.Objective(expr=1) # Set up a "dummy" objective


## Equality constraints (setting up the equation) ##
def _zdot(m, i):
    return m.dzdt[i] == m.z[i]**2 - 2*m.z[i] + 1
model.z_dot = pyo.Constraint(model.points, rule=_zdot)


def _back_diff(m, i):
    # If we are at the first datapoint, there is no previous datapoint 
    # to refer to so skip constraint
    if i == 0:
        return pyo.Constraint.Skip
    return m.dzdt[i] == (m.z[i] - m.z[i-1]) / m.h
model.back_diff = pyo.Constraint(model.points, rule=_back_diff)


def _init_con(m):
    return m.z[0] == -3
model.init_con = pyo.Constraint(rule=_init_con)


## Solving the model ##
solver = pyo.SolverFactory('ipopt')
solver.solve(model, tee=True)

## Comparing with analytical solution and plotting ##
import matplotlib.pyplot as plt

analytical_t = [0.01*i for i in range(0,101)] # Note: 0.01 hard-coded but in principle just similar to h used in model
analytical_z = [(4*t-3) / (4*t+1) for t in analytical_t]

finitediff_t = [model.h*i for i in model.points]
finitediff_z = [pyo.value(model.z[i]) for i in model.points]

plt.plot(analytical_t, analytical_z, 'b', label='analytical sol')
plt.plot(finitediff_t, finitediff_z, 'ro-', label='finite diff sol')
plt.legend(loc='best')
plt.xlabel('t')
plt.show()