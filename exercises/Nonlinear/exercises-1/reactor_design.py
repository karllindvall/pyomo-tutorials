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

# Known values
ca_in = 10000.0              # gmol m^-3
k_1 = 5.0 / 6.0             # min^-1
k_2 = 5.0 / 3.0             # min^-1
k_3 = 1.0 / 6000.0          # m^3 / mol*min


# Initialize model
model = pyo.ConcreteModel()


# Define variables
model.ca = pyo.Var(initialize=5000.0, within=pyo.PositiveReals)
model.cb = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
model.cc = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
model.cd = pyo.Var(initialize=1000.0, within=pyo.PositiveReals)
model.sv = pyo.Var(initialize=1.0, within=pyo.PositiveReals)


# Set up objective
model.obj = pyo.Objective(expr=model.cb, sense=pyo.maximize)


# Set up constraints
def ca_mol_bal_rule(m):
    return 0 == m.sv * ca_in - m.sv * m.ca - k_1 * m.ca - 2 * k_3 * m.ca ** 2
model.ca_mol_bal = pyo.Constraint(rule=ca_mol_bal_rule)

def cb_mol_bal_rule(m):
    return 0 == - m.sv * m.cb + k_1 * m.ca - k_2 * m.cb
model.cb_mol_bal = pyo.Constraint(rule=cb_mol_bal_rule)

def cc_mol_bal_rule(m):
    return 0 == - m.sv * m.cc + k_2 * m.cb
model.cc_mol_bal = pyo.Constraint(rule=cc_mol_bal_rule)

def cd_mol_bal_rule(m):
    return 0 == - m.sv * m.cd + k_3 * m.ca ** 2
model.cd_mol_bal = pyo.Constraint(rule=cd_mol_bal_rule)


# Choose solver and solve model
pyo.SolverFactory('ipopt').solve(model)
print(pyo.value(model.cb))