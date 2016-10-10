from __future__ import division, print_function, absolute_import

from keras.layers import Input, merge, Dense, TimeDistributed, Dropout, BatchNormalization
from keras import models

from .layers import NeuralGraphHidden, NeuralGraphOutput

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
							conv_bias=True, fp_bias=True, net_bias=True,
							conv_dropout=0, fp_dropout=0, net_dropout=0,
							conv_batchnorm=False, fp_batchnorm=False, net_batchnorm=False,
							atomwise_dropout=True, input_fp_out=True):
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
		conv_batchnorm, fp_batchnorm, net_batchnorm: Wheter or not to use
			batchnormalisation in these layers
		conv_dropout, fp_dropout, net_dropout: Dropout values in these layers
		atomwise_dropout: Should dropout of atom features be atom wise or not
			(only applies to conv and fp layers).
		input_fp_out: wether or not to add a `NeuralGraphOutput` layer that
			connects directely to the features

	# Returns:
		output: Ouput of final layer of network. Add prediciton layer and use
			functional API to turn into a model

	# NOTE:
		Dropout can be performd atom-wise or batch-wise. Another sensible option
			would be weight-wise dropout. In order for this to work the layers
			would need to support multiple layers (#5) and keras would need
			support as well (fchollet/keras/#3995).

	'''

	atoms, bonds, edges = data_input

	def ConvDropout(p_dropout):
		''' Defines the standard Dropout layer for convnets
		'''
		if atomwise_dropout:
			return TimeDistributed(Dropout(p_dropout))
		return Dropout(p_dropout)

	# Add first output layer directly to atom inputs
	if input_fp_out:
		fp_out = NeuralGraphOutput(fp_length, activation=fp_activation, bias=fp_bias)([atoms, bonds, edges])
		if fp_batchnorm:
			fp_out = BatchNormalization()(fp_out)

	# Add Graph convolutional layers
	convolved_atoms = [atoms]
	fingerprint_outputs = [fp_out]
	for conv_width in conv_layer_sizes:

		# Add hidden layer
		atoms_in = convolved_atoms[-1]
		if conv_dropout:
			atoms_in = ConvDropout(conv_dropout)(atoms_in)
		atoms_out = NeuralGraphHidden(conv_width, activation=conv_activation, bias=conv_bias)([atoms_in, bonds, edges])
		if conv_batchnorm:
			atoms_out = BatchNormalization()(atoms_out)

		# Add output layer
		fp_atoms_in = atoms_out
		if fp_dropout:
			fp_atoms_in = ConvDropout(fp_dropout)(fp_atoms_in)	
		fp_out = NeuralGraphOutput(fp_length, activation=fp_activation, bias=fp_bias)([fp_atoms_in, bonds, edges])
		if fp_batchnorm:
			fp_out = BatchNormalization()(fp_out)

		# Export
		convolved_atoms.append(atoms_out)
		fingerprint_outputs.append(fp_out)

	# Merge fingerprint
	final_fp = merge(fingerprint_outputs, mode='sum')
	
	# Add regular Neural net
	net_outputs = [final_fp]
	for layer_size in net_layer_sizes:

		# Add regular nn layers
		net_in = net_outputs[-1]
		if net_dropout:
			net_in = ConvDropout(net_dropout)(net_in)	
		net_out = Dense(layer_size, activation=net_activation, bias=net_bias)(net_in)
		if fp_batchnorm:
			net_out = BatchNormalization()(net_out)

		# Export
		net_outputs.append(net_out)

	return net_outputs[-1]