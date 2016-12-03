''' Classes for sparse vectors, lists of related tensors and tensors describing
molecular graphs
'''
from __future__ import division, print_function, absolute_import

import numpy as np
import pickle as pkl

from .utils import mol_dims_to_shapes, mol_shapes_to_dims

class SparseTensor(object):
    ''' An immutable class for sparse tensors of any shape, type and sparse value.

    # Arguments
        nonsparse_indices (nested int-array): List of arrays with indices for
            nonsparse elements at each dimension
        nonsparse_values (int-array): array of corresponding values
        default_value (of same dtype): The value that will be used for the non-
            sparse indices
        dtype (str/np.dtype): dtype, if `None`, dtype of nonsparse_values will be
            used
        main_axis (int): Axis along which `len` and `__getitem__ ` will work.
        assume sorted (bool): Only set to true if `nonsparse_indices[main_axis]`
            is sorted! (To speed up initialisation)

    # Attributes
        shape (tuple): The sparse tensor has no real shape, `tensor.as_array()`
            takes a `shape` argument. However, the tensor does have a mimimum
            size for each dimension (determined by the nonsparse element at the
            furthest position on that dimension)
        dtype (str/dtype): Can be changed after the tensor is created
        ndims (int): number of dimensions

    # Notes
        - This class is optimised for storage of data. The idea is that one of the
            dimensions is declared to be the `main_axis`. (This would be the axis
            along which the different datapoints are defined). All indexing occurs
            along this axis.
        - This class is not optimised for tensor operations, use `as_array` / numpy
            for that
        - Is best initialised trough the classmethod `from_array`
        - As the object is like an immutable object, there is no support for
            assignment or retrieval of individual entries. Use
                `tensor.as_array()[indices]` instead.
        - Currently, code is mainly optimised for retrieval of (relatively small)
            batches.

    # TODO, possible optimisations:
        - discard main index but use lookup when storing
        - build new lookup during __getitem__ and pass on init of new tensor to
            avoid expensive rebuilding
    '''
    def __init__(self, nonsparse_indices, nonsparse_values, default_value=0,
                 max_shape=None, dtype=None, main_axis=0, assume_sorted=False):

        # Assert valid index and convert negative indices to positive
        ndims = len(nonsparse_indices)
        main_axis = range(ndims)[main_axis]

        self.main_axis = main_axis
        self.default_value = default_value

        # Sort if necessary
        if not assume_sorted and len(nonsparse_values):
            nonsparse_entries = zip(nonsparse_values, *nonsparse_indices)
            sorted(nonsparse_entries, key=lambda x: x[main_axis+1])
            sorted_entries = zip(*nonsparse_entries)
            nonsparse_values = list(sorted_entries[0])
            nonsparse_indices = list(sorted_entries[1:])

        self.nonsparse_indices = [np.array([]) for _ in range(ndims)]
        self.nonsparse_values = np.array([])

        # Convert indices and values to numpy array and check dimensionality
        for i, ind in enumerate(nonsparse_indices):
            assert len(ind) == len(nonsparse_values), 'nonsparse_indices (size{0} @index {1}) should be of same length as nonsparse_values (size {2})'.format(len(ind), i, len(nonsparse_values))
            nonsparse_indices[i] = np.array(ind, dtype='int')
            self.nonsparse_indices = nonsparse_indices
        self.nonsparse_values = np.array(nonsparse_values)

        # Calculate and set the shape
        if len(self.nonsparse_values):
            self.true_shape = tuple([max(inds)+1 for inds in nonsparse_indices])
        else:
            self.true_shape = tuple([0]*ndims)

        # Setting dtype will alter self.nonsparse_values
        dtype = dtype or self.nonsparse_values.dtype
        self.dtype = dtype

        # Setting max_shape will check if shape matches with nonsparse entries
        self.max_shape = max_shape or [None]*ndims

        # Build lookup for quick indexing along the main_axis
        #   lookup defines first position of that element
        self.lookup = np.searchsorted(nonsparse_indices[self.main_axis],
                                      range(self.shape[self.main_axis]+1))

    @property
    def max_shape(self):
        return self._max_shape

    @max_shape.setter
    def max_shape(self, max_shape):
        for true_s, max_s, in zip(self.true_shape, max_shape):
            assert (max_s is None) or (max_s>=true_s) , 'Cannot set max_shape {} smaller than true shape {}'.format(max_shape, self.true_shape)
        self._max_shape = tuple(max_shape)

    @property
    def shape(self):
        return tuple([true_s if max_s==None else max_s
                     for true_s, max_s in zip(self.true_shape, self.max_shape)])

    @property
    def ndims(self):
        return len(self.nonsparse_indices)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = np.dtype(dtype)
        self.nonsparse_values = self.nonsparse_values.astype(self.dtype)

    def _nonsparse_entries(self, keys):
        ''' Returns indices and values required to create a new SparseTensor
            given the provided keys (along main_axis)

        # Arguments:
            keys (int/list): The keys for which to return the nonspare entries

        # Returns:
            indices (np.array): the new nonsparse indices (concatenated)
            values (np.array): the corresponding values (concatenated)

        # Note:
            mainly meant for internal use. Helper function of `self.__getitem__`

        '''
        if isinstance(keys, int):

            while keys < 0:
                keys += len(self)

            start_stop = self.lookup[keys:keys+2]
            if len(start_stop):
                inds = range(*start_stop)
            else:
                inds = []

            indices = [indices[inds] for indices in self.nonsparse_indices]
            values = self.nonsparse_values[inds]

            return indices, values

        elif isinstance(keys, (list, tuple, np.ndarray)):

            indices = [[] for _ in range(self.ndims)]
            values = []

            for g, key in enumerate(keys):
                add_indices, add_values = self._nonsparse_entries(key)
                values.append(add_values)
                for i in range(self.ndims):
                    if i == self.main_axis:
                        # For the main_axis, rewrite the keys in chronological
                        #   order (e.g. respect the ordering provided by keys)
                        indices[i].append(np.array([g]*len(add_values)))
                    else:
                        indices[i].append(add_indices[i])

            indices = [np.concatenate(inds) for inds in indices]
            values = np.concatenate(values)

            return indices, values

        else:
            raise ValueError

    # Magic funcions
    def __len__(self):
        return self.shape[self.main_axis]

    def __getitem__(self, keys):
        '''Gets the requested datapoints (along main axis) as SparseTensor

        # Arguments:
            keys (int, slice, list-like): Only one dimensional indexing is allowed

        # Returns:
            tensor (selfDataTensor): A new `SparseTensor` that corresponds
                to the requested keys
        '''

        # Ensure keys is of usable type
        if isinstance(keys, slice):
            start, stop, step = keys.indices(len(self))
            keys = range(start, stop, step)
        if isinstance(keys, (tuple, list, np.ndarray)):
            if len(keys) == 0:
                raise IndexError('Cannot index `SparseTensor` with empty slice (`[]`)')
            else:
                assert isinstance(keys[0], int), 'Indexing is only allowed along the main axis ({})'.format(self.main_axis)
        elif isinstance(keys, int):
            pass
        else:
            raise IndexError('Only int, list, np.ndarray or slice (`:`) allowed for indexing `SparseTensor`')

        # Copy properties of self to be passed to child object (make them mutatable)
        indices, values = self._nonsparse_entries(keys)
        max_shape = list(self.max_shape)
        main_axis = int(self.main_axis)

        # If getting a single element, drop singleton dimension
        if isinstance(keys, int):
            indices.pop(main_axis)
            max_shape.pop(main_axis)
            # Determining the new main axis is actually a trivial decision
            main_axis = min(main_axis, len(max_shape)-1)

        return self.__class__(dtype=self.dtype,
                              nonsparse_indices=indices, nonsparse_values=values,
                              main_axis=main_axis, default_value=self.default_value,
                              max_shape=max_shape)

    def __repr__(self):
        return "%s(dtype='%s', nonsparse_indices=%r, nonsparse_values=%r, main_axis=%r, default_value=%r, max_shape=%r)" % (
                self.__class__.__name__, self.dtype,
                [list(ind) for ind in self.nonsparse_indices],
                list(self.nonsparse_values), self.main_axis, self.default_value,
                self.max_shape)

    def __str__(self):
        return "%s(dtype='%s', shape=%s, default_value=%s)" % (
                self.__class__.__name__, self.dtype, self.shape, self.default_value)

    def __eq__(self, other):
        ''' Returns true if the sparse matrix can be expressed as other (by
        forcing it into the same shape).

        If shapes cannot match, raises

        Note that `sparse.as_array(full_shape) == sparse` will have good performance,
            because it uses this method, but `sparse == sparse.as_array(full_shape)`
            will not work well, because numpy (1.11.2) will try to do the comparison
            instead of calling this method.
        '''

        if isinstance(other, SparseTensor):
            other = other.as_array()
            shape = [max(s, o) for s,o in zip(self.shape, other.shape)]
        else:
            other = np.array(other)
            shape = other.shape

        return self.as_array(shape) == other

    def __ne__(self, other):
        return np.invert(self == other)

    # Export and import functionality
    @classmethod
    def from_array(cls, arr, dtype=None, main_axis=0, default_value=0,
                   max_shape=None):
        ''' Turns a regular array or array-like into a SparseTensor

        # Arguments:
            arr (array-like): The array to convert into a SparseTensor
            dtype (str/np.dtype): The datatype to use. If none is provided then
                `np.array(arr).dtype` will be used
            default_value (of same dtype): The nonsparse value to filter out

        # Returns:
            tensor (SparseTensor): s.t. `tensor.as_array(arr.shape) == arr`

        '''

        arr = np.array(arr)

        nonsparse_indices = list(np.where(arr != default_value))
        nonsparse_values = arr[nonsparse_indices]

        # Assume_sorted if main_axis=0 because of np.where
        assume_sorted = main_axis==0

        return cls(dtype=arr.dtype, nonsparse_indices=nonsparse_indices,
                   nonsparse_values=nonsparse_values, main_axis=0,
                   max_shape=max_shape, default_value=default_value,
                   assume_sorted=assume_sorted)


    def as_array(self, shape=None):
        '''Returns the SparseTensor as a nonsparse np.array

        # Arguments:
            shape (tuple/list): option to overwrite `self.max_shape` for
                this call. Array returned will have this shape.

                If None, `self.shape` will be used. (note that `self.shape` is
                defined by `self.max_shape`, or `self.true_shape` where `self.max_shape`
                is None). None values can also be used for individual dimensions
                wihin the shape tuple/list.

                Note that `shape` should be at least as big as `self.true_shape`.

        # Returns:
            out (np.array): nonsparse array of self.dtype

        '''

        if not shape:
            shape = [None] * self.ndims

        # Overwrite None with self.shape
        shape = [true_s if s==None else s for s, true_s in zip(shape, self.shape)]
        # Check if obtained shape matches with self.true_shape
        assert np.all([s >=true_s for s, true_s in zip(shape, self.true_shape)]), 'shape ({}) should be at least {}'.format(shape, self.true_shape)

        out = np.zeros(shape, dtype=self.dtype)
        out.fill(self.default_value)
        out[self.nonsparse_indices] = self.nonsparse_values

        return out

    def to_config(self, jsonify=False):
        ''' Returns a dict that can be used to recreate the file efficiently

        # Arguments:
            jsonify (bool): If True, dict will be jsonifiably (no `np.arrays`)

        # Returns:
            config (dict): that can be used in `SparseTensor.from_config`

        '''
        if jsonify:
            nonsparse_indices = [i.tolist() for i in self.nonsparse_indices]
            nonsparse_values = self.nonsparse_values.tolist()
        else:
            nonsparse_indices = self.nonsparse_indices
            nonsparse_values = self.nonsparse_values

        return dict(nonsparse_indices=nonsparse_indices, nonsparse_values=nonsparse_values,
                    default_value=self.default_value, dtype=str(self.dtype),
                    main_axis=self.main_axis, max_shape=self.max_shape,)

    @classmethod
    def from_config(cls, config):
        ''' Returns a SparseTensor based on the `config` dict
        '''
        return cls(nonsparse_indices=config['nonsparse_indices'],
                    nonsparse_values=config['nonsparse_values'],
                    default_value=config['default_value'], dtype=config['dtype'],
                    main_axis=config['main_axis'], max_shape=config['max_shape'],
                    assume_sorted=True)

class TensorList(object):
    ''' Helperclass to cluster tensors together, acts as a single list by propageting
    calls and slicing trough it's members.

    # Arguments:
        tensors (list of iterables): Should have the same length

    # Example:
        ```
        >>> tensors = TensorList([np.zeros((5,4)), np.ones((5,2,2)), -np.ones((5,))])
        >>> tensors.shape
        [(5, 4), (5, 2, 2), (5,)]
        >>> tensors[0:1]
        [array([[ 0.,  0.,  0.,  0.]]), array([[[ 1.,  1.], [ 1.,  1.]]]), array([-1.])]
        ```
    '''

    def __init__(self, tensors):
        lengths = set([len(t) for t in tensors])
        assert len(lengths) == 1, 'Length of all tensors should be the same'
        self.length = list(lengths)[0]
        self.tensors = tensors

    def map(self, fn):
        ''' Apply function to all tensors and return result
        '''
        return [fn(t) for t in self.tensors]

    def apply(self, fn):
        ''' Apply function to all tensors and replace with
        '''
        self.tensors = self.map(fn)

    def __getitem__(self, key):
        return [t[key] for t in self.tensors]

    @property
    def shape(self):
        return [t.shape for t in self.tensors]

    def __repr__(self):
        return "%s(tensors=%r)" % (self.__class__.__name__, self.tensors)

    def __len__(self):
        return self.length

class GraphTensor(TensorList):
    ''' Datacontainer for (molecular) graph tensors.

    This datacontainer mainly has advantages for indexing. The three tensors
        describing the graph are grouped in a tensorlist so that `graph_tensor[x]`
        will return atoms[x], bonds[x], edges[x]

    Furthermore, this container allows for sparse dimensions. A sparse dimension
        means that for each batch, that dimension is minimized to the maximum
        length that occurs within that batch.

    # Arguments:
        mol_tensors (tuple): tuple of np.array of nonspares mol tensors
            (atoms, bonds, edges)
        sparse_max_atoms (bool):  Wether or not max_atoms should be a sparse
            dimension.
        sparse_max_degree (bool): Wether or not max_degree should be a sparse
            dimension.

    '''
    def __init__(self, mol_tensors, sparse_max_atoms=True, sparse_max_degree=False):

        self.sparse_max_atoms = sparse_max_atoms
        self.sparse_max_degree = sparse_max_degree

        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_molecules) = mol_shapes_to_dims(mol_tensors)

        # Set sparse dimension sizes to None
        num_molecules = None
        if sparse_max_atoms:
            max_atoms = None
        if sparse_max_degree:
            max_degree = None

        max_shapes = mol_dims_to_shapes(max_atoms, max_degree, num_atom_features,
                                       num_bond_features)

        # Convert into SparseTensors
        atoms, bonds, edges = mol_tensors
        atoms = SparseTensor.from_array(atoms, max_shape=max_shapes[0])
        bonds = SparseTensor.from_array(bonds, max_shape=max_shapes[1])
        edges = SparseTensor.from_array(edges, max_shape=max_shapes[2], default_value=-1)

        # Initialise with result
        super(GraphTensor, self).__init__([atoms, bonds, edges])

    def __getitem__(self, keys):

        # Make sure we don't lose the num_molecules dimension
        if isinstance(keys, int):
            keys = [keys]

        # Get each sliced tensor as a new `SparseTensor` object
        sliced_tensors = [t[keys] for t in self.tensors]

        # Make sure that max_atoms and max_degree match across all tensors,
        #   (for isolated nodes (atoms), this is not always the case)
        # Use the max value across all tensors
        max_atoms_vals = [t.shape[1] for t in sliced_tensors]
        max_degree_vals = [t.shape[2] for t in sliced_tensors[1:]]

        max_atoms = max(max_atoms_vals)
        max_degree = max(max_degree_vals)

        # Return tensors with the matching shapes
        shapes = mol_dims_to_shapes(max_atoms, max_degree, None, None, len(keys))
        return [t.as_array(shape) for t, shape in zip(sliced_tensors, shapes)]

    @property
    def max_shape(self):
        return [t.max_shape for t in self.tensors]

    @property
    def true_shape(self):
        return [t.max_shape for t in self.tensors]

class EpochIterator(object):
    ''' Iterates over a dataset. (designed for keras fit_generator)

    # Arguments:
        data (tuple): Tuple of data to iterate trough, usually `(x_data, y_data)`,
            though a tuple of any length can be passed. The iterables inside the
            tuple should support list-indexing.
        batch_size (int): Number of datapoints yielded per batch
        epochs (int/None): maximum number of epochs after which a `StopIteration`
            is raised (None for infinite generator)
        shuffle (bool): Wether to shuffle at the onset of each epoch

    # Yields
        batch (tuple): tuple corresponding to the `data` tuple that contains a
            slice of length `batch_size` (except possibly on last batch of epoch)

    # Example
        using `keras.models.model`
        >>> model.fit_generator(EpochIterator(np.array(zip(data, labels)))

    # Note
        designed for use with keras `model.fit_generator`
    '''
    def __init__(self, data, batch_size=1, epochs=None, shuffle=True):
        self.data = TensorList(data)
        self.epochs = epochs or np.inf
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Initialise counters
        self.reset()

    def __iter__(self):
        return self

    def next(self):
        # At the end of an epoch, raise Stopiteration, or reset counter
        if self.i >= len(self.data):
            if self.epoch >= self.epochs:
                raise StopIteration
            else:
                self.i = 0
                self.epoch += 1

        # At the begin of an epoch, shuffle the order of the data
        if self.i==0 and self.shuffle:
            np.random.shuffle(self.indices)

        # Get the indices for this batch, and update counter i
        use_inds = self.indices[self.i:self.i+self.batch_size]
        self.i += len(use_inds)

        # Return as tuple
        return tuple(self.data[use_inds])

    def reset(self):
        ''' Resets the counters of the iterator
        '''
        self.i = 0
        self.epoch = 1
        self.indices = range(len(self.data))


def unit_tests_sparse_tensor(seed=None):

    np.random.seed(seed)

    arr = np.random.randint(3, size=(2000,30,5,8))
    sparse = SparseTensor.from_array(arr)

    singleton_shape = arr.shape[1:]
    full_shape = (None,) + singleton_shape

    print('Testing: `as_array` should return same as input to `from_array`')
    assert np.all(sparse.as_array(full_shape) == arr)

    print('Testing: Integer indexing should be identical to numpy')
    assert np.all(sparse[0].as_array(singleton_shape) == arr[0])

    print('Testing: Negative integer indexing should be identical to numpy')
    assert np.all(sparse[len(sparse)-1].as_array(singleton_shape) == sparse[-1].as_array(singleton_shape) )

    print('Testing: List indexing should be identical to numpy')
    get_inds = [2,-1,3,6,0,0,1]
    assert np.all(sparse[get_inds].as_array(full_shape) == arr[get_inds])

    print('Testing: Slice indexing should be identical to numpy')
    assert np.all(sparse[::-1].as_array(full_shape) == arr[::-1])

    print('Testing: Various indexing testcases that should return same array as sparse')
    assert np.all(sparse.as_array(full_shape) == sparse[:].as_array(full_shape))
    assert np.all(sparse.as_array(full_shape) == sparse[0:len(sparse)+10].as_array(full_shape))

    print('Testing: Equality functions return `True` for all entries when comparing sparse with sparse')
    assert np.all(sparse == sparse.as_array(full_shape))
    # assert np.all(sparse.as_array(full_shape) == sparse)

    print('Testing: Equality functions return `True` for all entries when comparing sparse with original array')
    assert np.all(arr == sparse.as_array(full_shape))
    # assert np.all(sparse.as_array(full_shape) == arr)

    print('Testing: Equality functions should return same boolean array as numpy')
    assert np.all((arr[0] == 0) == (sparse[0] == 0))
    assert np.all((arr[0] == arr[3]) == (sparse[0] == sparse[3]))

    print('Testing: Inequality functions return `False` for all entries when comparing sparse with sparse')
    assert not np.all(sparse != sparse.as_array(full_shape))
    # assert not np.all(sparse.as_array(full_shape) != sparse)

    print('Testing: Inequality functions return `False` for all entries when comparing sparse with original array')
    assert not np.all(arr != sparse.as_array(full_shape))
    assert not np.all(sparse.as_array(full_shape) != arr)

    print('Testing: Ineuality functions should return same boolean array as numpy')
    assert np.all((arr[0] != 0) == (sparse[0] != 0))
    assert np.all((arr[0] != arr[3]) == (sparse[0] != sparse[3]))

    print('Testing: `repr` can reproduce sparse')
    assert np.all(eval(repr(sparse)) == sparse)

    print('Testing: `from_config` can reproduce `sparse.to_config`')
    assert np.all(SparseTensor.from_config(sparse.to_config(False)) == sparse)
    assert np.all(SparseTensor.from_config(sparse.to_config(True)) == sparse)

    print('Testing: unpickled pickles object reproduces itself')
    assert np.all(pkl.loads(pkl.dumps(sparse)) == sparse)
    assert np.all(pkl.loads(pkl.dumps(sparse)) == sparse.as_array())

def unit_tests_graph_tensor(seed=None):
    np.random.seed(seed)

    # Parameters for generative model
    num_molecules=50
    max_atoms = 40
    max_degree = 6
    num_atom_features = 62
    num_bond_features = 8

    # Generate/simulate graph tensors
    atoms = np.zeros((num_molecules, max_atoms, num_atom_features))
    bonds = np.zeros((num_molecules, max_atoms, max_degree, num_bond_features))
    edges = np.zeros((num_molecules, max_atoms, max_degree)) -1

    # Generate atoms for each molecule
    for i, n_atoms in  enumerate(np.random.randint(1, max_atoms, size=num_molecules)):
        atoms[i, 0:n_atoms, :] = np.random.randint(3, size=(n_atoms, num_atom_features))

        # Generator neighbours/bonds for each atom
        for a, degree in enumerate(np.random.randint(max_degree, size=n_atoms)):
            bonds[i, a, 0:degree, :] = np.random.randint(3, size=(degree, num_bond_features))
            edges[i, a, 0:degree] = np.random.randint(max_degree, size=degree)

    mols = GraphTensor([atoms, bonds, edges], sparse_max_atoms=True,
                       sparse_max_degree=True)

    max_atoms_sizes = set([])
    max_degree_sizes = set([])
    num_atom_features_sizes = set([])
    num_bond_features_sizes = set([])
    num_molecules_sizes = set([])

    for i in range(len(mols)):
        # This asserts the shapes match within the tensors
        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_molecules) = mol_shapes_to_dims(mols[i])

        max_atoms_sizes.add(max_atoms)
        max_degree_sizes.add(max_degree)
        num_atom_features_sizes.add(num_atom_features)
        num_bond_features_sizes.add(num_bond_features)
        num_molecules_sizes.add(num_molecules)

    print('Testing: max_atoms is varying in size')
    assert len(max_atoms_sizes) > 1

    print('Testing: max_degree is varying in size')
    assert len(max_degree_sizes) > 1

    print('Testing: num_atom_features is constant in size')
    assert len(num_atom_features_sizes) == 1

    print('Testing: num_bond_features is constant in size')
    assert len(num_bond_features_sizes) == 1

    print('Testing: num_molecules is constant in size')
    assert len(num_molecules_sizes) == 1

def unit_test_epoch_iterator(seed=None):

    np.random.seed(seed)

    n_datapoints = 50
    batch_size = 13
    epochs = 100

    x_data = np.random.rand(n_datapoints, 3, 6, 2)
    y_data = np.random.rand(n_datapoints, 8)

    it = EpochIterator((x_data, y_data), epochs=epochs, batch_size=batch_size)

    x_lengths = []
    y_lengths = []
    epochs = []
    for x, y in it:
        x_lengths.append(len(x))
        y_lengths.append(len(y))
        epochs.append(it.epoch)

    x_lengths = np.array(x_lengths)
    y_lengths = np.array(y_lengths)

    seen = x_lengths.cumsum()
    true_epoch1 = np.floor(seen / n_datapoints).astype(int)
    true_epoch2 = np.array([0] + list(true_epoch1[:-1]))
    iter_epochs = np.array(epochs) - epochs[0]

    print('Testing: x and y lengths match')
    assert np.all(x_lengths == y_lengths)

    print('Testing: epoch are correct size')
    assert np.all(iter_epochs == true_epoch1) or np.all(iter_epochs == true_epoch2)


def unit_tests(seed=None):
    print("\n{:=^100}".format(' Unit tests for `SparseTensor` '))
    unit_tests_sparse_tensor(seed=seed)

    print("\n{:=^100}".format('  Unit tests for `GraphTensor`  '))
    unit_tests_graph_tensor(seed=seed)

    print("\n{:=^100}".format('  Unit tests for `EpochIterator`  '))
    unit_test_epoch_iterator(seed=seed)
    print('All unit tests passed!')

if __name__ == '__main__':
    unit_tests()