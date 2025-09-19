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


### Setting up model and packages ###

m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)

# Property package for benzene, toluene, hydrogen, methane mixture
m.fs.BTHM_params = HDAParameterBlock()
# Prop package for benzene-toluene mixture
m.fs.BT_params = BTXParameterBlock(
    valid_phase=("Liq", "Vap"), activity_coeff_model="Ideal"
)
# Rxn package for HDA rxn
m.fs.reaction_params = reaction_props.HDAReactionParameterBlock(
    property_package=m.fs.BTHM_params
)

# ------------------------------------------------------------------------------------------- #

### Adding unit models to flowsheet ###

# Mixer 101
m.fs.M101 = Mixer(
    property_package=m.fs.BTHM_params,
    inlet_list=["toluene_feed", "hydrogen_feed", "vapor_recycle"],
)
# Heater 101
m.fs.H101 = Heater(property_package=m.fs.BTHM_params, has_phase_equilibrium=True)
# CSTR R101
m.fs.R101 = CSTR(
    property_package=m.fs.BTHM_params,
    reaction_package=m.fs.reaction_params,
    has_heat_of_reaction=True,
    has_heat_transfer=True,
    )
# Flash 101
m.fs.F101 = Flash(
    property_package=m.fs.BTHM_params,
    has_heat_transfer=True,
    has_pressure_change=True,
)
# Splitter 101
m.fs.S101 = Splitter(
    property_package=m.fs.BTHM_params,
    outlet_list=["purge", "recycle"],
)
# Pressure changer C101
m.fs.C101 = PressureChanger(
    property_package=m.fs.BTHM_params,
    compressor=True,
    thermodynamic_assumption=ThermodynamicAssumption.isothermal,
)

# ---------------------------------------------------------------------------------------------- #

# Translater block to convert between property packages
m.fs.translator = Translator(
    inlet_property_package=m.fs.BTHM_params,
    outlet_property_package=m.fs.BT_params,
)
# Must add constraints to translator block so 
# it knows how to translate between property packages!
# Constraints 1: Total flow = benzene flow + toluene flow (molar)
m.fs.translator.eq_total_flow = Constraint(
    expr=m.fs.translator.outlet.flow_mol[0]
    == m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "benzene"] # index: [time, phase, component]
    + m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "toluene"], # valid phases: liq, vap
)
# Constraint 2: Outlet temp = inlet temp
m.fs.translator.eq_temperature = Constraint(
    expr=m.fs.translator.outlet.temperature[0]
    == m.fs.translator.inlet.temperature[0]
)
# Constraint 3: Outlet pressure = inlet pressure
m.fs.translator.eq_pressure = Constraint(
    expr=m.fs.translator.outlet.pressure[0]
    == m.fs.translator.inlet.pressure[0]
)
# Constraint 4: Benzene mol frac definition
m.fs.translator.eq_mole_frac_benzene = Constraint(
    expr=m.fs.translator.outlet.mole_frac_comp[0, "benzene"]
    == m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "benzene"]
    / (
        m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "benzene"]
        + m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "toluene"]
    )
)
# Constraint 5: Toluene mol frac definition
m.fs.translator.eq_mole_frac_toluene = Constraint(
    expr=m.fs.translator.outlet.mole_frac_comp[0, "toluene"]
    == m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "toluene"]
    / (
        m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "toluene"]
        + m.fs.translator.inlet.flow_mol_phase_comp[0, "Liq", "benzene"]
    )
)

# ---------------------------------------------------------------------------------------- #

# Adding Heater 102 with new thermodynamic package
m.fs.H102 = Heater(
    property_package=m.fs.BT_params,
    has_pressure_change=True,
    has_phase_equilibrium=True,
)

# ---------------------------------------------------------------------------------------- #

### Connecting individual units using Arc ###

m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.H101.inlet)
m.fs.s04 = Arc(source=m.fs.H101.outlet, destination=m.fs.R101.inlet)
m.fs.s05 = Arc(source=m.fs.R101.outlet, destination=m.fs.F101.inlet)
m.fs.s06 = Arc(source=m.fs.F101.vap_outlet, destination=m.fs.S101.inlet)
m.fs.s08 = Arc(source=m.fs.S101.recycle, destination=m.fs.C101.inlet)
m.fs.s09 = Arc(source=m.fs.C101.outlet, destination=m.fs.M101.vapor_recycle)
m.fs.s10a = Arc(source=m.fs.F101.liq_outlet, destination=m.fs.translator.inlet)
m.fs.s10b = Arc(source=m.fs.translator.outlet, destination=m.fs.H102.inlet)

# Writing equality constraints between the two points
TransformationFactory("network.expand_arcs").apply_to(m)

# Adding additional constraint on conversion in R101
# Define conversion variable
m.fs.R101.conversion = Var(initialize=0.75, bounds=(0,1))
# Append constraint to the model
m.fs.R101.conv_constraint = Constraint(
    expr=m.fs.R101.conversion * m.fs.R101.inlet.flow_mol_phase_comp[0, "Vap", "toluene"]
    == (
        m.fs.R101.inlet.flow_mol_phase_comp[0, "Vap", "toluene"]
        - m.fs.R101.outlet.flow_mol_phase_comp[0, "Vap", "toluene"]
    )
)

# ---------------------------------------------------------------------------------------- #

### Fixing conditions ###

# Pure toluene stream (others components assigned tiny values for convergence)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-5)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-5)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(1e-5)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "methane"].fix(1e-5)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-5)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "toluene"].fix(0.30)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-5)
m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-5)
m.fs.M101.toluene_feed.temperature.fix(303.2)
m.fs.M101.toluene_feed.pressure.fix(350000)

# Hydrogen stream
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-5)
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-5)
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(0.30)
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "methane"].fix(0.02)
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-5)
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "toluene"].fix(1e-5)
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-5)
m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-5)
m.fs.M101.hydrogen_feed.temperature.fix(303.2)
m.fs.M101.hydrogen_feed.pressure.fix(350000)

### Fixing unit model specifications ###

# H101
m.fs.H101.outlet.temperature.fix(600)
# R101
m.fs.R101.conversion.fix(0.75)
m.fs.R101.heat_duty.fix(0)
# F101
m.fs.F101.vap_outlet.temperature.fix(325.0)
m.fs.F101.deltaP.fix(0)
# S101
m.fs.S101.split_fraction[0, "purge"].fix(0.2)
# C101
m.fs.C101.outlet.pressure.fix(350000)
# H102
m.fs.H102.outlet.temperature.fix(375)
m.fs.H102.deltaP.fix(-200000) # pressure drop

# Scaling factors applied to avoid convergence issues
iscale.set_scaling_factor(m.fs.H101.control_volume.heat, 1e-2)
iscale.set_scaling_factor(m.fs.R101.control_volume.heat, 1e-2)
iscale.set_scaling_factor(m.fs.R101.control_volume.rate_reaction_extent, 1)
iscale.set_scaling_factor(m.fs.R101.control_volume.volume, 1)
iscale.set_scaling_factor(m.fs.F101.control_volume.heat, 1e-2)
iscale.set_scaling_factor(m.fs.H102.control_volume.heat, 1e-2)

# Scaling factors for remaining vars and all constraints
iscale.calculate_scaling_factors(m.fs.H101)
iscale.calculate_scaling_factors(m.fs.R101)
iscale.calculate_scaling_factors(m.fs.F101)
iscale.calculate_scaling_factors(m.fs.H102)

# ---------------------------------------------------------------------------------------- #

### Initializing flowsheet ###

seq = SequentialDecomposition()
seq.options.select_tear_method = "heuristic"
seq.options.tear_method = "Wegstein"
seq.options.iterLim = 3

G = seq.create_graph(m)
heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
order = seq.calculation_order(G)

def SeqDecompInfo():
    """Made this function so that these outputs 
    aren't displayed every time the code is run."""
    # Check tear stream (fs.s03)
    for o in heuristic_tear_set:
        print(o.name)
    # Check solving sequence w/ least number tears
    for o in order:
        print(o[0].name)
    return

# Give initial guess for tear stream - assume 0 recycle
tear_guesses = {
    "flow_mol_phase_comp": {
        (0, "Vap", "benzene"): 1e-5,
        (0, "Vap", "toluene"): 1e-5,
        (0, "Vap", "hydrogen"): 0.30,
        (0, "Vap", "methane"): 0.02,
        (0, "Liq", "benzene"): 1e-5,
        (0, "Liq", "toluene"): 0.30,
        (0, "Liq", "hydrogen"): 1e-5,
        (0, "Liq", "methane"): 1e-5,
    },
    "temperature": {0: 303.2},
    "pressure": {0: 350000},
}

# Pass in our tear guess
seq.set_guesses_for(m.fs.H101.inlet, tear_guesses)

# Initialization of units
def function(unit):
    # Try to initialize using default
    # if this fails (probably due to scaling) try for the second time
    try:
        initializer = unit.default_initializer()
        initializer.initialize(unit, output_level=idaeslog.INFO)
    except InitializationError:
        solver = get_solver()
        solver.solve(unit)

# Initializing in sequence mode
# Only set iterLim to 3 because we only want set of initial values for IPOPT to take over later
seq.run(m, function)

# Run flowsheet in simulation mode
solver = get_solver()
results = solver.solve(m, tee=True)

# ---------------------------------------------------------------------------------------- #

### Distillation column ###

# SequentialDecomposition does not support distillation column model, so
# we add the distillation column after decomposition and then simulate 
# the entire flowsheet.

m.fs.D101 = TrayColumn(
    number_of_trays=10,
    feed_tray_location=5,
    condenser_type=CondenserType.totalCondenser,
    condenser_temperature_spec=TemperatureSpec.atBubblePoint,
    property_package=m.fs.BT_params
)

# Connect H102 outlet to D101 inlet
m.fs.s11 = Arc(source=m.fs.H102.outlet, destination=m.fs.D101.feed)

# Add equality constraints
TransformationFactory("network.expand_arcs").apply_to(m)

# Propagate the state
propagate_state(m.fs.s11)

# Fix properties
m.fs.D101.condenser.reflux_ratio.fix(0.5)
m.fs.D101.reboiler.boilup_ratio.fix(0.5)
m.fs.D101.condenser.condenser_pressure.fix(150000)

# Set scaling factors
iscale.set_scaling_factor(m.fs.D101.condenser.control_volume.heat, 1e-2)
iscale.set_scaling_factor(m.fs.D101.reboiler.control_volume.heat, 1e-2)

iscale.calculate_scaling_factors(m.fs.D101)

# Initialize distillation column
m.fs.D101.initialize(outlvl=idaeslog.INFO)

# Solve entire flowsheet
solver.solve(m, tee=True)

# After solving, get the states of streams entering and leaving a certain block:
m.fs.R101.report()
m.fs.F101.report()

# Create a basic stream table
from idaes.core.util.tables import (
    create_stream_table_dataframe,
    stream_table_dataframe_to_string
)

st = create_stream_table_dataframe({"Reactor": m.fs.s05, "Light Gases": m.fs.s06})
print(stream_table_dataframe_to_string(st))

# For optimization of a certain parameter, start by defining the objective
# m.fs.objective = Objective(expr=...)

# Then unfix decision variables so the system has more than 0 DOF
# m.fs.H101.outlet.temperature.unfix()
# m.fs.R101.conversion.unfix()
# etc...

# Next, set bounds on decision variables
# m.fs.H101.outlet.temperature[0].setlb(500)
# m.fs.H101.outlet.temperature[0].setub(600)
# etc...

# Lastly, define constraints (desired optimization conditions) and re-solve model
# m.fs.product_flow = Constraint(expr=m.fs.D101.condenser.distillate.flow_mol[0] >= 0.18)
# m.fs.product_purity = Constraint(expr=m.fs.D101.condenser.distillate.mole_frac_comp[0, "benzene"] >= 0.99)
# results = solver.solve(m, tee=True)