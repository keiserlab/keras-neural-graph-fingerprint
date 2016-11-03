from __future__ import division, print_function, absolute_import

from keras.layers import Input, merge, Dense
from keras import models

import utils
from NGF.preprocessing import tensorise_smiles, tensorise_smiles_mp
from NGF.layers import NeuralGraphHidden, NeuralGraphOutput
from NGF.models import build_graph_conv_model
from NGF.sparse import SparseTensor, TensorList, EpochIterator

# ==============================================================================
# ================================ Load the data ===============================
# ==============================================================================
print("{:=^100}".format(' Data preprocessing '))
data, labels = utils.load_delaney()

# Tensorise data
X_atoms, X_bonds, X_edges = tensorise_smiles_mp(data)
print('Atoms:', X_atoms.shape)
print('Bonds:', X_bonds.shape)
print('Edges:', X_edges.shape)

# Load sizes from data shape
num_molecules = X_atoms.shape[0]
max_atoms = X_atoms.shape[1]
max_degree = X_bonds.shape[2]
num_atom_features = X_atoms.shape[-1]
num_bond_features = X_bonds.shape[-1]

# ==============================================================================
# =============== Example 1: Building a 3-layer graph convnet  =================
# ==============================================================================
print("{:=^100}".format(' Example 1 '))

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

# ==============================================================================
# ============ Example 2: Initialising layers in different ways  ===============
# ==============================================================================
print("{:=^100}".format(' Example 2 '))

# Parameters
conv_width = 8
fp_length = 62

# Define the input layers
atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')

# Define the convoluted atom feature layers
# All methods of initialisation are equaivalent!
atoms1 = NeuralGraphHidden(lambda: Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
atoms2 = NeuralGraphHidden(Dense(conv_width, activation='relu', bias=False))([atoms1, bonds, edges])

# Define the outputs of each (convoluted) atom featuer layer to fingerprint
# All methods of initialisation are equaivalent!
fp_out0 = NeuralGraphOutput(Dense(fp_length, activation='softmax'))([atoms0, bonds, edges])
fp_out1 = NeuralGraphOutput(fp_length, activation='softmax')([atoms1, bonds, edges])
fp_out2 = NeuralGraphOutput(lambda: Dense(fp_length, activation='softmax'))([atoms2, bonds, edges])

# Sum outputs to obtain fingerprint
final_fp = merge([fp_out0, fp_out1, fp_out2], mode='sum')

# Build and compile model for regression.
main_prediction = Dense(1, activation='linear', name='main_prediction')(final_fp)
model2 = models.Model(input=[atoms0, bonds, edges], output=[main_prediction])
model2.compile(optimizer='adagrad', loss='mse')

# Show summary
model2.summary()

# ==============================================================================
# ================== Example 3: Using the model functions  =====================
# ==============================================================================
print("{:=^100}".format(' Example 3 '))

model3 = build_graph_conv_model(max_atoms, num_atom_features, num_bond_features, max_degree,
								learning_type='regression', fp_length=fp_length,
								conv_layer_sizes=[conv_width, conv_width],
								conv_activation='relu', fp_activation='softmax',
								conv_bias=False)

# Show summary
model3.summary()

# ==============================================================================
# ===================== Example 4: Using sparse tensors  =======================
# ==============================================================================
print("{:=^100}".format(' Example 4 '))
# Using sparse tensors will improve training speed a lot, because the number of
#   max_atoms will be determined by the number molecule with the most atoms within
#   a batch, rather than the molecule with the most atoms within the dataset

# Build the same model, but this time use None for num_atom_features, to allow
#   variation of this variable per batch.
model4 = build_graph_conv_model(None, num_atom_features, num_bond_features, max_degree,
                                learning_type='regression', fp_length=fp_length,
                                conv_layer_sizes=[conv_width, conv_width],
                                conv_activation='relu', fp_activation='softmax',
                                onv_bias=False)

# Show summary
model4.summary()

# Convert the atom features into sparse tensors, use return_array=True so that once
#   we take a sample from the indes, they are returned as a np.array.

# The sparse tensor subsample will have the smallest dimensions possible,
#   for num_atoms and max_atoms, this is a desired effect, however, for max_degree
#   and num_bond_features, we will set a max_shape, to controll their size
sparse_atoms = SparseTensor.from_array(X_atoms, max_shape=(None, None, num_atom_features), return_array=True)
sparse_bonds = SparseTensor.from_array(X_bonds, max_shape=(None, None, max_degree, num_bond_features), return_array=True)
sparse_edges = SparseTensor.from_array(X_edges, max_shape=(None, None, max_degree), return_array=True, default_value=-1)

# Merge (sparse) features and data into a tensorlist, so that taking a slice of
#   the list will return a list of the sliced tensors.
X_mols = TensorList([sparse_atoms, sparse_bonds, sparse_edges])
data = TensorList([X_mols, labels])

# Build a generator and train the model
my_generator = EpochIterator(data, batch_size=128)
model4.fit_generator(my_generator, nb_epoch=20, samples_per_epoch=len(labels), pickle_safe=True)
