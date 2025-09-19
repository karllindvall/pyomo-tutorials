'Building a neural network formulation from scratch using OMLT'

import numpy as np
import pyomo.environ as pyo

from omlt.neuralnet import NetworkDefinition
from omlt.neuralnet.layer import DenseLayer, IndexMapper, InputLayer

# Creating NN definition with input = 4 vars, 2x2 grid with bounds [-1, 1]
net = NetworkDefinition(
    scaled_input_bounds={
        (0, 0): (-1.0, 1.0),
        (0, 1): (-1.0, 1.0),
        (1, 0): (-1.0, 1.0),
        (1, 1): (-1.0, 1.0),
    }
)

# Adding input layer (2x2 matrix)
input_layer = InputLayer([2, 2])
net.add_layer(input_layer)

# Adding dense layer
dense_layer_0 = DenseLayer(
    input_size=input_layer.output_size,
    output_size=[2, 1],
    activation="linear",
    weights=np.array([[1.0], [-0.5]]),
    biases=np.array([[0.1], [0.25]]),
)
net.add_layer(dense_layer_0)
net.add_edge(input_layer, dense_layer_0)

# Adding last layer used to flatten output into vector with same # elements
transformer = IndexMapper([2, 1], [2])
dense_layer_1 = DenseLayer(
    input_size=[2],
    output_size=[1],
    activation="relu",
    weights=np.array([[2.0], [-1.0]]),
    biases=np.array([[0.0]]),
    input_index_mapper=transformer,
)
net.add_layer(dense_layer_1)
net.add_edge(dense_layer_0, dense_layer_1)


# Building optimization model
from omlt import OmltBlock
from omlt.neuralnet import ReluBigMFormulation

m = pyo.ConcreteModel()

# OmltBlock automatically builds optimization formulation of an ML model
m.neural_net = OmltBlock()
# Instantiate formulation object (associates NN with formulation)
formulation = ReluBigMFormulation(net)
# Generate formulation as we specified (i.e., NeuralNetworkFormulation)
m.neural_net.build_formulation(formulation)


# Exploring formulation

# OMLT blocks have indexed vars representing block input and outputs
# Shows var locations, bounds, value, etc.
m.neural_net.inputs.pprint()
m.neural_net.outputs.pprint()

# Layer defines a set of vars 'zhat' representing Yhat (output before activation) 
# and a set of vars 'z' representing output after activation (Y).
# Notes:
#       1) Can also index by layer's name: m.neural_net.layer['input_layer'].z.pprint()
#       2) .at(i) grabs ith element 1-based so if i=1, we are looking at input layer
m.neural_net.layer[m.neural_net.layers.at(1)].z.pprint()
m.neural_net.layer[m.neural_net.layers.at(2)].zhat.pprint()

# Block connects input vars with input layer output vars
m.neural_net.input_assignment.pprint()

# We can also see the assignment that Y = Yhat, representing linear activation fxn
m.neural_net.layer[m.neural_net.layers.at(2)].linear_activation.pprint()

# Looking at the dense layer with ReLU

m.neural_net.layer[m.neural_net.layers.at(3)].z.pprint()
m.neural_net.layer[m.neural_net.layers.at(3)].zhat.pprint()
m.neural_net.layer[m.neural_net.layers.at(3)].dense_layer.pprint()

# Output Y cannot be assigned directly to Yhat b/c activation fxn is non-linear
# We define and compute bounds on Yhat, and represent ReLU as mixed integer programming (MIP)
# Y >= 0; Y >= Yhat; Y <= Yhat(@ upper bound)*delta; Y <= Yhat - Yhat(@ lower bound)*(1-delta)
# with delta a binary: 1 when activation fxn is in 0 region, otherwise 0.

# Looking at lower and upper bounds
m.neural_net.layer[m.neural_net.layers.at(3)]._big_m_lb_relu.pprint()
m.neural_net.layer[m.neural_net.layers.at(3)]._big_m_ub_relu.pprint()

# Can also print MIP constraints
m.neural_net.layer[m.neural_net.layers.at(3)]._z_lower_bound_relu.pprint()
m.neural_net.layer[m.neural_net.layers.at(3)]._z_lower_bound_zhat_relu.pprint()
m.neural_net.layer[m.neural_net.layers.at(3)]._z_upper_bound_relu.pprint()
m.neural_net.layer[m.neural_net.layers.at(3)]._z_upper_bound_zhat_relu.pprint()

# Finally, output of model where outputs of last layer are constrained to outputs vars
m.neural_net.output_assignment.pprint()

# Layers can also be evaulated with a given input IF it is a numpy array of size layer.input_size.
x = np.diag([1.0, 0.1]) + 0.3
x_inp = input_layer.eval_single_layer(x) # input=output for this layer
x_dl0 = dense_layer_0.eval_single_layer(x) # Y = WX + b operation & returns 2x1 np matrix
x_dl1 = dense_layer_1.eval_single_layer(x) # Y = WX + B w/ different weights, return vector w/ 1 element