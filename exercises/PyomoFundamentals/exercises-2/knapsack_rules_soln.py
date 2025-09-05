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


import pyomo.environ as pyo

A = ['hammer', 'wrench', 'screwdriver', 'towel']
b = {'hammer':8, 'wrench':3, 'screwdriver':6, 'towel':11}
w = {'hammer':5, 'wrench':7, 'screwdriver':4, 'towel':3}
W_max = 14

model = pyo.ConcreteModel()
model.x = pyo.Var( A, within=pyo.Binary )

def obj_rule(m):
    return sum( b[i]*m.x[i] for i in A )
model.obj = pyo.Objective(rule=obj_rule, sense = pyo.maximize )

def weight_con_rule(m):
    return sum( w[i]*m.x[i] for i in A ) <= W_max
model.weight_con = pyo.Constraint(rule=weight_con_rule)

opt = pyo.SolverFactory('glpk')

model.int_cuts = pyo.ConstraintList()

for k in range(5):
    # Solve the current model
    result_obj = opt.solve(model)

    # Print sol
    output_str = "Obj: " + str(pyo.value(model.obj))
    for i in A:
        output_str += "  x[%s]: %f" % (str(i), pyo.value(model.x[i]))
    print(output_str)

    # Add cut so same solution is not provided twice
    cut = 0
    for i in A:
        x_i = round(pyo.value(model.x[i]))
        cut += (1 - model.x[i]) if x_i == 1 else model.x[i]
    model.int_cuts.add(cut >= 1)
