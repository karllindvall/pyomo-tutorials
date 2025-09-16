from pyomo.environ import (
    Constraint,
    Var,
    ConcreteModel,
    Expression,
    Objective,
    TransformationFactory,
    value
)
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import (
    PressureChanger,
    Mixer,
    Separator as Splitter,
    Heater,
    CSTR,
    Flash,
    Translator
)
from idaes.models_extra.column_models import TrayColumn
from idaes.models_extra.column_models.condenser import CondenserType, TemperatureSpec

# Utility tools to put together flowsheet and calculate DOFs
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from idaes.core.util.exceptions import InitializationError

# Logger to set output levels
import idaes.logger as idaeslog

# Thermo and reaction packages
from hda_files import hda_reaction as reaction_props
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
from hda_files.hda_ideal_VLE import HDAParameterBlock

