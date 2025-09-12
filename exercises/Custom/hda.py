import pyomo.environ as pyo
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import PressureChanger, Mixer, Separator, Splitter, Heater, StoichiometricReactor, Flash
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import InitializationError

# To set output levels
import idaes.logger as idaeslog

model = pyo.ConcreteModel()
model.fs = FlowsheetBlock(dynamic=False)

# Adding property package
model.fs.thermo_params = 