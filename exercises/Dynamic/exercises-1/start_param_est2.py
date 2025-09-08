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
from pyomo.dae import ContinuousSet, DerivativeVar

a_conc = {0.1:0.606, 0.2:0.368, 0.3:0.223, 0.4:0.135, 0.5:0.082,
          0.6:0.05, 0.7:0.03, 0.8:0.018, 0.9:0.011, 1.0:0.007}

b_conc = {0.1:0.373, 0.2:0.564, 0.3:0.647, 0.4:0.669, 0.5:0.656,
          0.6:0.624, 0.7:0.583, 0.8:0.539, 0.9:0.494, 1.0:0.451}

m = pyo.ConcreteModel()

m.meas_time = pyo.Set(initialize=sorted(a_conc.keys()),ordered=True)
m.ameas = pyo.Param(m.meas_time, initialize=a_conc)
m.bmeas = pyo.Param(m.meas_time, initialize=b_conc)

m.time = ContinuousSet(initialize=m.meas_time, bounds=(0,1))

## Initialize variables rxn rate, conc, dconc/dt
m.k1 = pyo.Var()
m.k2 = pyo.Var()

m.a = pyo.Var(m.time, bounds=(0,1))
m.b = pyo.Var(m.time, bounds=(0,1))

m.dadt = DerivativeVar(m.a)
m.dbdt = DerivativeVar(m.b)


## Set constraints
def _init_conditions(m):
    yield m.a[0] == 1
    yield m.b[0] == 0
m.init_cond = pyo.ConstraintList(rule=_init_conditions)

def _dca_condition(m, t):
    return m.dadt[t] == -m.k1 * m.a[t]
m.dca_cond = pyo.Constraint(m.time, rule=_dca_condition)

def _dcb_condition(m, t):
    return m.dbdt[t] == m.k1*m.a[t] - m.k2*m.b[t]
m.dcb_cond = pyo.Constraint(m.time, rule=_dcb_condition)

## Set objective
def _obj(m):
    return sum((m.a[t] - m.ameas[t])**2 + (m.b[t] - m.bmeas[t])**2 for t in m.meas_time)
m.obj = pyo.Objective(expr=_obj)


## Initialize discretizer
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=10, ncp=3, scheme="LAGRANGE-RADAU")


## Solve system
solver = pyo.SolverFactory('ipopt')
solver.solve(m, tee=True)


## Extracting pyomo variables and plotting
print('k1= '+str(pyo.value(m.k1)))
print('k2= '+str(pyo.value(m.k2)))

meas_time = list(m.meas_time)
a_meas = [pyo.value(m.ameas[i]) for i in m.meas_time]
b_meas = [pyo.value(m.bmeas[i]) for i in m.meas_time]

t = list(m.time)
a = [pyo.value(m.a[i]) for i in m.time]
b = [pyo.value(m.b[i]) for i in m.time]
    
import matplotlib.pyplot as plt

plt.plot(t,a,label='A')
plt.plot(t,b,label='B')
plt.plot(meas_time,a_meas,'o')
plt.plot(meas_time,b_meas,'o')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('concentration')
plt.title('Kinetic Parameter Estimation')
plt.show()