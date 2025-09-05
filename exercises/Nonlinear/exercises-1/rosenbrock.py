#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2023-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


# rosenbrock_soln.py
import pyomo.environ as pyo

model = pyo.ConcreteModel()
model.x = pyo.Var()
model.y = pyo.Var()

def rosenbrock(model):
    return (1.0-model.x)**2 \
        + 100.0*(model.y - model.x**2)**2
model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

y_val = 1.5
for i in range(2,6):
    model.x = i
    model.y = y_val
    pyo.SolverFactory('ipopt').solve(model, tee=True)

    print("{0:6.2f}  {1:6.2f}  {2:6.2f}  {3:6.2f}".format(i, \
            y_val, pyo.value(model.x), pyo.value(model.y)))
    