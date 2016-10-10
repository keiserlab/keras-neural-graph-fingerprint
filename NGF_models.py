from __future__ import division, print_function, absolute_import

from keras.layers import Input, merge, Dense
from keras import models

from NGF_layers import NeuralGraphHidden, NeuralGraphOutput

def build_graph_conv_model(max_atoms, num_atom_features, num_bond_features,
								max_degree, learning_type, 
								output_size=1, optimizer='adagrad',
								**kwargs):
	''' Builds and compiles a graph convolutional network with a regular neural
		network on top for regression.

	Especially usefull when using the sklearn `KerasClassifier` wrapper

	# Arguments
		max_atoms, num_atom_features, num_bond_features, max_degree: The
			dimensionalities used to create input layers.
		learning_type: Intended use of model, affects loss function and final 
			activation function used. allowed: 'regression', 'binary_class',
			'multi_class'
		output_size: size of prediciton layer
		optimizer: used to compile the model
		kwargs: Used to call `build_graph_conv_net`

	# Returns:
		keras Model: compiled for learning_type

	'''

	# Define the input layers
	atoms = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
	bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
	edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')

	# Get output of main net
	net_output = build_graph_conv_net([atoms, bonds, edges], **kwargs)

	# Add final prediction layer
	learning_type = learning_type.lower()
	if learning_type == 'regression':
		final_activation = 'linear'
		loss='mse'
	elif learning_type == 'binary_class':
		final_activation = 'sigmoid'
		loss='binary_crossentropy'
	elif learning_type == 'multi_class':
		final_activation = 'softmax'
		loss='categorical_crossentropy'
	else:
		raise Exception('Invalid argument for learning type ({})'.format(learning_type))
	main_prediction = Dense(output_size, activation=final_activation, name='main_prediction')(net_output)

	# Build and compile the model
	model = models.Model(input=[atoms, bonds, edges], output=[main_prediction])
	model.compile(optimizer=optimizer, loss=loss)		

	return model


def build_graph_conv_net(data_input,
							fp_length=1024, conv_layer_sizes=[], net_layer_sizes=[], 
							conv_activation='relu', fp_activation='softmax', net_activation='relu',
							conv_bias=True, fp_bias=True, net_bias=True):
	''' Builds a graph convolutional network with a regular neural network on
		top.

	# Arguments
		data_input: The Input feature layers (as `[atoms, bonds, edges]`)
		fp_length: size of fingerprint outputs, that will be feed into the 
			regular neural net on top of the graph net.
		conv_layer_sizes, net_layer_sizes: List of number of nodes in each hidden
			layer of the graph convolutional net, and the neural net resp.
		conv_activation, fp_activation, net_activation: Activitaion function used
			in the `NeuralGraphHidden`, `NeuralGraphOutput` and `Dense` layers
			respectively.
		conv_bias, fp_bias, net_bias: Wheter or not to use bias in these layers

	# Returns:
		output: Ouput of final layer of network. Add prediciton layer and use
			functional API to turn into a model

	# TODO:
		add dropout and batchnorm
	'''

	atoms, bonds, edges = data_input

	# Add first output layer directly to atom inputs
	fp_out = NeuralGraphOutput(fp_length, activation=fp_activation, bias=fp_bias)([atoms, bonds, edges])

	# Add Graph convolutional layers
	convolved_atoms = [atoms]
	fingerprint_outputs = [fp_out]
	for conv_width in conv_layer_sizes:

		atoms_out = NeuralGraphHidden(conv_width, activation=conv_activation, bias=conv_bias)([convolved_atoms[-1], bonds, edges])
		fp_out = NeuralGraphOutput(fp_length, activation=fp_activation, bias=fp_bias)([atoms_out, bonds, edges])

		convolved_atoms.append(atoms_out)
		fingerprint_outputs.append(fp_out)

	# Merge fingerprint
	final_fp = merge(fingerprint_outputs, mode='sum')
	
	# Add regular Neural net
	net_outputs = [final_fp]
	for layer_size in net_layer_sizes:
		net_out = Dense(layer_size, activation=net_activation, bias=net_bias)(net_outputs[-1])
		net_outputs.append(net_out)

	return net_outputs[-1]