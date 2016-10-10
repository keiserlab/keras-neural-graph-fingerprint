from __future__ import division, print_function, absolute_import

from keras.layers import Input, merge, Dense
from keras import models

import utils
from NGF.preprocessing import tensorise_smiles
from NGF.layers import NeuralGraphHidden, NeuralGraphOutput
from NGF.models import build_graph_conv_model


# ================================ Load the data ===============================
data, labels = utils.load_delaney()

# Tensorise data
X_atoms, X_bonds, X_edges = tensorise_smiles(data)
print('Atoms:', X_atoms.shape)
print('Bonds:', X_bonds.shape)
print('Edges:', X_edges.shape)

# Load sizes from data shape
num_molecules = X_atoms.shape[0]
max_atoms = X_atoms.shape[1]
max_degree = X_bonds.shape[2]
num_atom_features = X_atoms.shape[-1]
num_bond_features = X_bonds.shape[-1]


# ====================== Build the Neural Graph Convnet. =======================
# Parameters
conv_width = 8
fp_length = 62

# Define the input layers
atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')

# Define the convoluted atom feature layers
atoms1 = NeuralGraphHidden(conv_width, activation='relu', bias=False)([atoms0, bonds, edges])
atoms2 = NeuralGraphHidden(conv_width, activation='relu', bias=False)([atoms1, bonds, edges])

# Define the outputs of each (convoluted) atom featuer layer to fingerprint
fp_out0 = NeuralGraphOutput(fp_length, activation='softmax')([atoms0, bonds, edges])
fp_out1 = NeuralGraphOutput(fp_length, activation='softmax')([atoms1, bonds, edges])
fp_out2 = NeuralGraphOutput(fp_length, activation='softmax')([atoms2, bonds, edges])

# Sum outputs to obtain fingerprint
final_fp = merge([fp_out0, fp_out1, fp_out2], mode='sum')

# Build and compile model for regression.
main_prediction = Dense(1, activation='linear', name='main_prediction')(final_fp)
model = models.Model(input=[atoms0, bonds, edges], output=[main_prediction])
model.compile(optimizer='adagrad', loss='mse')

# Show summary
model.summary()

# Train the model
model.fit([X_atoms, X_bonds, X_edges], labels, nb_epoch=20, batch_size=32, validation_split=0.2)


# ============ Build the exact same net using the model function. ==============
model2 = build_graph_conv_model(max_atoms, num_atom_features, num_bond_features, max_degree, 
								learning_type='regression', fp_length=fp_length,
								conv_layer_sizes=[conv_width, conv_width],
								conv_activation='relu', fp_activation='softmax',
								conv_bias=False)

# Show summary
model2.summary()

# Train the model
model2.fit([X_atoms, X_bonds, X_edges], labels, nb_epoch=20, batch_size=32, validation_split=0.2)
