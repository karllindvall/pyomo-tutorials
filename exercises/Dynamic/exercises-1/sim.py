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
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator

m = pyo.ConcreteModel()

m.time = ContinuousSet(bounds=(0,1))

## Initialize variables rxn rate, conc, dconc/dt
m.k1 = pyo.Var(initialize=5)
m.k2 = pyo.Var(initialize=1)

m.a = pyo.Var(m.time)
m.b = pyo.Var(m.time)

m.dadt = DerivativeVar(m.a)
m.dbdt = DerivativeVar(m.b)

m.a[0].fix(1) # We now provide the initial values as "fixed" values instead of as initial conditions in a constraint
m.b[0].fix(0)

## Set constraints
def _dca_condition(m, t):
    return m.dadt[t] == -m.k1 * m.a[t]
m.dca_cond = pyo.Constraint(m.time, rule=_dca_condition)

def _dcb_condition(m, t):
    return m.dbdt[t] == m.k1*m.a[t] - m.k2*m.b[t]
m.dcb_cond = pyo.Constraint(m.time, rule=_dcb_condition)

import matplotlib.pyplot as plt


# Instead of solving, we want to simulate
mysim = Simulator(m, package='scipy')
tsim, profiles = mysim.simulate(integrator='vode', numpoints=100)
varorder = mysim.get_variable_order()
for idx, v in enumerate(varorder):
    plt.plot(tsim, profiles[:, idx], label=v)

plt.show()