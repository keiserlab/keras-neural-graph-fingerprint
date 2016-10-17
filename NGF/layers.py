from __future__ import print_function
import inspect

from keras import layers
from keras.utils.layer_utils import layer_from_config

import keras.backend as K

import theano

def filter_func_args(fn, args, invalid_args=[], overrule_args=[]):
    '''Separate a dict of arguments into one that a function takes, and the rest

    # Arguments:
        fn: arbitrary function
        args: dict of arguments to separate
        invalid_args: list of arguments that will be removed from args
        overrule_args: list of arguments that will be returned in other_args,
            even if they are arguments that `fn` takes

    # Returns:
        fn_args, other_args: tuple of separated arguments, ones that the function
            takes, and the others (minus `invalid_args`)
    '''

    fn_valid_args = inspect.getargspec(fn)[0]
    fn_args = {}
    other_args = {}
    for arg, val in args.iteritems():
        if not arg in invalid_args:
            if (arg in fn_valid_args) and (arg not in overrule_args):
                fn_args[arg] = val
            else:
                other_args[arg] = val
    return fn_args, other_args

# TODO: Rewrite this function to Keras and drop theano dependency
def parallel_gather(references, indices):
    '''
    Executes theano index (i.e. K.gather()) for each sample in a batch, usefull
    when dealing with Tensor of dim > 2D
    '''
    result, _ = theano.scan(fn=lambda reference, indices:reference[indices],
        outputs_info=None, sequences=[references, indices])
    return result

class NeuralGraphHidden(layers.Layer):
    ''' Hidden Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
    2015). This layer takes a graph as an input. The graph is represented as by
    three tensors.

    - The atoms tensor represents the features of the nodes.
    - The bonds tensor represents the features of the edges.
    - The edges tensor represents the connectivity (which atoms are connected to
        which)

    It returns the convolved features tensor, which is very similar to the atoms
    tensor. Instead of each node being represented by a num_atom_features-sized
    vector, each node now is represented by a convolved feature vector of size
    conv_width.

    # Example
        Define the input:
        ```python
            atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
            bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
            edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
        ```

        The `NeuralGraphHidden` can be initialised in three ways:
        1. Using an integer `conv_width` and possible kwags (`Dense` layer is used)
            ```python
            atoms1 = NeuralGraphHidden(conv_width, activation='relu', bias=False)([atoms0, bonds, edges])
            ```
        2. Using an initialised `Dense` layer
            ```python
            atoms1 = NeuralGraphHidden(Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
            ```
        3. Using a function that returns an initialised `Dense` layer
            ```python
            atoms1 = NeuralGraphHidden(lambda: Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
            ```

        Use `NeuralGraphOutput` to convert atom layer to fingerprint

    # Arguments
        inner_layer_arg: Either:
            1. an int defining the `conv_width`, with optional kwargs for the
                inner Dense layer
            2. An initialised but not build (`Dense`) keras layer (like a wrapper)
            3. A function that returns an initialised keras layer.
        kwargs: For initialisation 1. you can pass `Dense` layer kwargs

    # Input shape
        List of Atom and edge tensors of shape:
        `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
          bond_features), (samples, max_atoms, max_degrees)]`
        where degrees referes to number of neighbours

    # Output shape
        New atom featuers of shape
        `(samples, max_atoms, conv_width)`

    # References
        - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)

    '''

    def __init__(self, inner_layer_arg, **kwargs):
        # Initialise based on one of the three initialisation methods

        # Case 1: Check if inner_layer_arg is conv_width
        if isinstance(inner_layer_arg, (int, long)):
            self.conv_width = inner_layer_arg
            dense_layer_kwargs, kwargs = filter_func_args(layers.Dense.__init__,
            kwargs, overrule_args=['name'])
            self.create_inner_layer_fn = lambda: layers.Dense(self.conv_width, **dense_layer_kwargs)

        # Case 2: Check if an initialised keras layer is given
        elif isinstance(inner_layer_arg, layers.Layer):
            assert inner_layer_arg.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.conv_width = inner_layer_arg.get_output_shape_for((None, None))
            # layer_from_config will mutate the config dict, therefore create a get fn
            def get_inner_layer_config():
                return { 'config': inner_layer_arg.get_config(),
                                     'class_name': inner_layer_arg.__class__.__name__}
            self.create_inner_layer_fn = lambda: layer_from_config(get_inner_layer_config())

        # Case 3: Check if a function is provided that returns a initialised keras layer
        elif callable(inner_layer_arg):
            example_instance = inner_layer_arg()
            assert isinstance(example_instance, layers.Layer), 'When initialising with a function, the function has to return a keras layer'
            assert example_instance.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.conv_width = example_instance.get_output_shape_for((None, None))
            self.create_inner_layer_fn = inner_layer_arg

        else:
            raise ValueError('NeuralGraphHidden has to be initialised with 1). int conv_widht, 2). a keras layer instance, or 3). a function returning a keras layer instance.')

        super(NeuralGraphHidden, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        atoms_shape, bonds_shape, edges_shape = inputs_shape
        num_samples = atoms_shape[0]
        max_atoms = atoms_shape[1]
        num_atom_features = atoms_shape[-1]
        num_bond_features = bonds_shape[-1]

        max_degree = edges_shape[2]

        self.max_degree = max_degree

        # Add the dense layers (that contain trainable params)
        #   (for each degree we convolve with a different weight matrix)
        self.trainable_weights = []
        self.inner_3D_layers = []
        for degree in range(max_degree):

            # Initialise inner layer, and rename it
            inner_layer = self.create_inner_layer_fn()
            inner_layer_type = inner_layer.__class__.__name__.lower()
            inner_layer.name = self.name + '_inner_' + inner_layer_type + '_' + str(degree)

            # Initialise TimeDistributed layer wrapper in order to parallelise
            #   dense layer across atoms (3D)
            inner_3D_layer_name = self.name + '_inner_timedistributed_' + str(degree)
            inner_3D_layer = layers.TimeDistributed(inner_layer, name=inner_3D_layer_name)

            # Build the TimeDistributed layer (which will build the Dense layer)
            inner_3D_layer.build((None, max_atoms, num_atom_features+num_bond_features))

            # Store inner_3D_layer and it's weights
            self.inner_3D_layers.append(inner_3D_layer)
            self.trainable_weights += inner_3D_layer.trainable_weights

    def call(self, inputs, mask=None):
        atoms, bonds, edges = inputs

        # Import dimensions
        num_samples = atoms._keras_shape[0]
        max_atoms = atoms._keras_shape[1]
        num_atom_features = atoms._keras_shape[-1]
        num_bond_features = bonds._keras_shape[-1]

        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = K.sum(K.not_equal(edges, -1), axis=-1, keepdims=True)

        # The lookup masking trick: We add 1 to all indices, converting the
        #   masking value of -1 to a valid 0 index.
        masked_edges = edges + 1
        # We then add a zerovector to the left of the lookup matrix,
        # note: this keras adds a zero vector to the left AND right, but that is
        #   okay for now
        masked_atom_features_lookup = K.temporal_padding(atoms, padding=1)

        # For each atom, look up the featues of it's neighbour
        neighbour_atom_features = parallel_gather(masked_atom_features_lookup, masked_edges)

        # Sum along degree axis to get summed neighbour features, also add the self
        summed_atom_features = K.sum(neighbour_atom_features, axis=-2) + atoms

        # Sum the edge features for each atom
        summed_bond_features = K.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        summed_features = K.concatenate([summed_atom_features, summed_bond_features], axis=-1)

        # For each degree we convolve with a different weight matrix
        new_features_by_degree = []
        for degree in range(self.max_degree):

            # Create mask for this degree
            atom_masks_this_degree = K.equal(atom_degrees, degree)

            # Multiply with hidden merge layer
            #   (use time Distributed because we are dealing with 2D input/3D for batches)
            # Add keras shape to let keras now the dimensions
            summed_features._keras_shape = (None, max_atoms, num_atom_features+num_bond_features)
            new_unmasked_features = self.inner_3D_layers[degree](summed_features)

            # Do explicit masking because TimeDistributed does not support masking
            new_masked_features = new_unmasked_features * atom_masks_this_degree

            new_features_by_degree.append(new_masked_features)

        # Finally sum the features of all atoms
        new_features = layers.merge(new_features_by_degree, mode='sum')

        return new_features

    def get_output_shape_for(self, inputs_shape):

        # Import dimensions
        atoms_shape, bonds_shape, edges_shape = inputs_shape
        num_samples = atoms_shape[0]
        max_atoms = atoms_shape[1]
        num_features = atoms_shape[2]
        max_degree = edges_shape[2]

        return (num_samples, max_atoms, self.conv_width)

    @classmethod
    def from_config(cls, config):
        # Use layer build function to initialise new NeuralHiddenLayer
        inner_layer_config = config.pop('inner_layer_config')
        create_inner_layer_fn = lambda: layer_from_config(inner_layer_config)

        layer = cls(create_inner_layer_fn, **config)
        return layer

    def get_config(self):
        config = super(NeuralGraphHidden, self).get_config()

        # Store config of (a) inner layer of the 3D wrapper
        inner_layer = self.inner_3D_layers[0].layer
        config['inner_layer_config'] = { 'config': inner_layer.get_config(),
                                        'class_name': inner_layer.__class__.__name__}
        return config


class NeuralGraphOutput(layers.Layer):
    ''' Output Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
    2015). This layer takes a graph as an input. The graph is represented as by
    three tensors.

    - The atoms tensor represents the features of the nodes.
    - The bonds tensor represents the features of the edges.
    - The edges tensor represents the connectivity (which atoms are connected to
        which)

    It returns the fingerprint vector for each sample for the given layer.

    According to the original paper, the fingerprint outputs of each hidden layer
    need to be summed in the end to come up with the final fingerprint.

    # Example
        Define the input:
        ```python
            atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
            bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
            edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
        ```

        The `NeuralGraphOutput` can be initialised in three ways:
        1. Using an integer `fp_length` and possible kwags (`Dense` layer is used)
            ```python
            fp_out = NeuralGraphOutput(fp_length, activation='relu', bias=False)([atoms0, bonds, edges])
            ```
        2. Using an initialised `Dense` layer
            ```python
            fp_out = NeuralGraphOutput(Dense(fp_length, activation='relu', bias=False))([atoms0, bonds, edges])
            ```
        3. Using a function that returns an initialised `Dense` layer
            ```python
            fp_out = NeuralGraphOutput(lambda: Dense(fp_length, activation='relu', bias=False))([atoms0, bonds, edges])
            ```

        Predict for regression:
        ```python
        main_prediction = Dense(1, activation='linear', name='main_prediction')(fp_out)
        ```

    # Arguments
        inner_layer_arg: Either:
            1. an int defining the `fp_length`, with optional kwargs for the
                inner Dense layer
            2. An initialised but not build (`Dense`) keras layer (like a wrapper)
            3. A function that returns an initialised keras layer.
        kwargs: For initialisation 1. you can pass `Dense` layer kwargs

    # Input shape
        List of Atom and edge tensors of shape:
        `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
          bond_features), (samples, max_atoms, max_degrees)]`
        where degrees referes to number of neighbours

    # Output shape
        Fingerprints matrix
        `(samples, fp_length)`

    # References
        - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)

    '''

    def __init__(self, inner_layer_arg, **kwargs):
        # Initialise based on one of the three initialisation methods

        # Case 1: Check if inner_layer_arg is fp_length
        if isinstance(inner_layer_arg, (int, long)):
            self.fp_length = inner_layer_arg
            dense_layer_kwargs, kwargs = filter_func_args(layers.Dense.__init__,
            kwargs, overrule_args=['name'])
            self.create_inner_layer_fn = lambda: layers.Dense(self.fp_length, **dense_layer_kwargs)

        # Case 2: Check if an initialised keras layer is given
        elif isinstance(inner_layer_arg, layers.Layer):
            assert inner_layer_arg.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.fp_length = inner_layer_arg.get_output_shape_for((None, None))
            self.create_inner_layer_fn = lambda: inner_layer_arg

        # Case 3: Check if a function is provided that returns a initialised keras layer
        elif callable(inner_layer_arg):
            example_instance = inner_layer_arg()
            assert isinstance(example_instance, layers.Layer), 'When initialising with a function, the function has to return a keras layer'
            assert example_instance.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.fp_length = example_instance.get_output_shape_for((None, None))
            self.create_inner_layer_fn = inner_layer_arg

        else:
            raise ValueError('NeuralGraphHidden has to be initialised with 1). int conv_widht, 2). a keras layer instance, or 3). a function returning a keras layer instance.')

        super(NeuralGraphOutput, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        atoms_shape, bonds_shape, edges_shape = inputs_shape
        max_atoms = atoms_shape[1]
        max_degree = edges_shape[2]
        num_atom_features = atoms_shape[-1]
        num_bond_features = bonds_shape[-1]

        # Add the dense layer that contains the trainable parameters
        # Initialise dense layer with specified params (kwargs) and name
        inner_layer = self.create_inner_layer_fn()
        inner_layer_type = inner_layer.__class__.__name__.lower()
        inner_layer.name = self.name + '_inner_'+ inner_layer_type

        # Initialise TimeDistributed layer wrapper in order to parallelise
        #   dense layer across atoms
        inner_3D_layer_name = self.name + '_inner_timedistributed'
        self.inner_3D_layer = layers.TimeDistributed(inner_layer, name=inner_3D_layer_name)

        # Build the TimeDistributed layer (which will build the Dense layer)
        self.inner_3D_layer.build((None, max_atoms, num_atom_features+num_bond_features))

        # Store dense_3D_layer and it's weights
        self.trainable_weights = self.inner_3D_layer.trainable_weights


    def call(self, inputs, mask=None):
        atoms, bonds, edges = inputs

        # Import dimensions
        num_samples = atoms._keras_shape[0]
        max_atoms = atoms._keras_shape[1]
        num_atom_features = atoms._keras_shape[-1]
        num_bond_features = bonds._keras_shape[-1]

        # Create a matrix that stores for each atom, the degree it is, use it
        #   to create a general atom mask (unused atoms are 0 padded)
        # We have to use the edge vector for this, because in theory, a convolution
        #   could lead to a zero vector for an atom that is present in the molecule
        atom_degrees = K.sum(K.not_equal(edges, -1), axis=-1, keepdims=True)
        general_atom_mask = K.not_equal(atom_degrees, 0)

        # Sum the edge features for each atom
        summed_bond_features = K.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        atoms_bonds_features = K.concatenate([atoms, summed_bond_features], axis=-1)

        # Compute fingerprint
        atoms_bonds_features._keras_shape = (None, max_atoms, num_atom_features+num_bond_features)
        fingerprint_out_unmasked = self.inner_3D_layer(atoms_bonds_features)

        # Do explicit masking because TimeDistributed does not support masking
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        # Sum across all atoms
        final_fp_out = K.sum(fingerprint_out_masked, axis=-2)

        return final_fp_out

    def get_output_shape_for(self, inputs_shape):

        # Import dimensions
        atoms_shape, bonds_shape, edges_shape = inputs_shape
        num_samples = atoms_shape[0]
        max_atoms = atoms_shape[1]
        num_features = atoms_shape[2]
        max_degree = edges_shape[2]

        return (num_samples, self.fp_length)

    @classmethod
    def from_config(cls, config):
        # Use layer build function to initialise new NeuralGraphOutput
        inner_layer_config = config.pop('inner_layer_config')
        create_inner_layer_fn = lambda: layer_from_config(inner_layer_config)

        layer = cls(create_inner_layer_fn, **config)
        return layer

    def get_config(self):
        config = super(NeuralGraphOutput, self).get_config()

        # Store config of inner layer of the 3D wrapper
        inner_layer = self.inner_3D_layer.layer
        config['inner_layer_config'] = { 'config': inner_layer.get_config(),
                                        'class_name': inner_layer.__class__.__name__}
        return config

class AtomwiseDropout(layers.Layer):
    ''' Performs dropout over an atom feature vector where each atom will get
    the same dropout vector.

    Eg. With an input of `(batch_n, max_atoms, atom_features)`, a dropout mask of
    `(batch_n, atom_features)` will be generated, and repeated `max_atoms` times

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    '''
    def __init__(self, p, **kwargs):
        self.dropout_layer = layers.Dropout(p)
        self.uses_learning_phase = self.dropout_layer.uses_learning_phase
        self.supports_masking = True
        super(AtomwiseDropout, self).__init__(**kwargs)

    def _get_noise_shape(self, x):
        return None

    def call(self, inputs, mask=None):
        # Import dimensions
        _, max_atoms, num_atom_features = inputs._keras_shape

        # By [farizrahman4u](https://github.com/fchollet/keras/issues/3995)
        ones = layers.Lambda(lambda x: (x * 0 + 1)[:, 0, :], output_shape=lambda s: (s[0], s[2]))(inputs)
        dropped = self.dropout_layer(ones)
        dropped = layers.RepeatVector(max_atoms)(dropped)
        return layers.Lambda(lambda x: x[0] * x[1], output_shape=lambda s: s[0])([inputs, dropped])

    def get_config(self):
        config = super(AtomwiseDropout, self).get_config()
        config['p'] = self.dropout_layer.p
        return config

#TODO: Add GraphWiseDropout layer, that creates masks for each degree separately.