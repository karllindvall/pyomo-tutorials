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

# Inlet conditions
model.fs.flash.inlet.flow_mol.fix(1)                            # Inlet overall mol flow = 1
model.fs.flash.inlet.temperature.fix(368)                       # Inlet T = 368 K
model.fs.flash.inlet.pressure.fix(101325)                       # Inlet p = 101325 Pa
model.fs.flash.inlet.mole_frac_comp[0, "benzene"].fix(0.5)      # Inlet mol frac benzene = 0.5
model.fs.flash.inlet.mole_frac_comp[0, "toluene"].fix(0.5)      # Inlet mol frac toluene = 0.5

# Flash unit specs
model.fs.flash.heat_duty.fix(0)                                 # Heat duty on flash = 0
model.fs.flash.deltaP.fix(0)                                    # No pressure drop across flash tank

# Initialization and allowing heat duty to vary to compensate for constraint
model.fs.flash.initialize()
model.fs.flash.heat_duty.unfix()

model.benz_mol_frac_con = pyo.Constraint(
    expr=model.fs.flash.vap_outlet.mole_frac_comp[0, "benzene"] == 0.6
)

solver = pyo.SolverFactory("ipopt")
status = solver.solve(model)

model.fs.flash.report()