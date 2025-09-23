import pyomo.environ as pyo
from idaes.core import FlowsheetBlock

# To set output levels
import idaes.logger as idaeslog

model = pyo.ConcreteModel()
model.fs = FlowsheetBlock(dynamic=False)

# Importing and creating properties block
# To define appropriate state vars and eqs for thermo calculations
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
model.fs.properties = BTXParameterBlock(
    valid_phase=("Liq", "Vap"), activity_coeff_model="Ideal", state_vars="FTPz"
)

# Import "Flash" unit model and create instance of unit
from idaes.models.unit_models import Flash
model.fs.flash = Flash(property_package=model.fs.properties)

# Check the DOFs
from idaes.core.util.model_statistics import degrees_of_freedom
# print(f"Degrees of freedom = {degrees_of_freedom(model)}")

# Need to specify 7 DOFs
# Inlet conditions
model.fs.flash.inlet.flow_mol.fix(1)                            # Inlet overall mol flow = 1
model.fs.flash.inlet.temperature.fix(368)                       # Inlet T = 368 K
model.fs.flash.inlet.pressure.fix(101325)                       # Inlet p = 101325 Pa
model.fs.flash.inlet.mole_frac_comp[0, "benzene"].fix(0.5)      # Inlet mol frac benzene = 0.5
model.fs.flash.inlet.mole_frac_comp[0, "toluene"].fix(0.5)      # Inlet mol frac toluene = 0.5

# Flash unit specs
model.fs.flash.heat_duty.fix(0)                                 # Heat duty on flash = 0
model.fs.flash.deltaP.fix(0)                                    # No pressure drop across flash tank

# print(f"Degrees of freedom after specs = {degrees_of_freedom(model)}")

# Initializing flash unit
model.fs.flash.initialize() # add outlvl=idaeslog.INFO for info initialization

# Solving the model
solver = pyo.SolverFactory("ipopt")
status = solver.solve(model, tee=True)


## Displaying results ##

# Current value of flash vapour outlet pressure
#print(f"Pressure = {pyo.value(model.fs.flash.vap_outlet.pressure[0])}")

# Showing vapour and liquid outlet info
#print()
#print("Output from display:")
#model.fs.flash.vap_outlet.display()
#model.fs.flash.liq_outlet.display()

# Report shows key variables in flash model, i.e., inlet, vapour, and liquid ports
#model.fs.flash.report()


## Implementing a varying heat duty ##

# First, need to write code for "solve_successful" from workshop tutorials
def solve_successful(status):
    if (
        status.solver.termination_condition == pyo.TerminationCondition.optimal
        or status.solver.status == pyo.SolverStatus.ok
    ):
        return True
    return False

import numpy as np

# Store results for plotting
Q = []
V = []

# Re-initialize model
model.fs.flash.initialize(outlvl=idaeslog.WARNING)

# For varying heat duties
for duty in np.linspace(-17000, 25000, 50):
    model.fs.flash.heat_duty.fix(duty)
    Q.append(duty)
    status = solver.solve(model)

    if solve_successful(status):
        V.append(pyo.value(model.fs.flash.vap_outlet.flow_mol[0]))
    else:
        V.append(0.0)
        print(f"Solve with {pyo.value(model.fs.flash.heat_duty[0])} failed.")

# Plotting
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax = plt.subplots()
    plt.plot(Q, V)
    plt.grid()
    ax.set(
        xlabel="Heat Duty [J]",
        ylabel="Vapour Fraction [-]",
        title="Vapour Fraction at varying heat duties."
    )
    plt.show()