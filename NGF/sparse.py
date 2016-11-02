from __future__ import division, print_function, absolute_import

import numpy as np

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
    '''
    def __init__(self, nonsparse_indices, nonsparse_values, default_value=0,
                 dtype=None, main_axis=0, assume_sorted=False):

        # Assert valid index and convert negative indices to positive
        ndims = len(nonsparse_indices)
        main_axis = range(ndims)[main_axis]

        self.main_axis = main_axis
        self.default_value = default_value

        # if not assume_sorted:
        if not assume_sorted and len(nonsparse_values):
            nonsparse_entries = zip(nonsparse_values, *nonsparse_indices)
            sorted(nonsparse_entries, key=lambda x: x[main_axis+1])
            sorted_entries = zip(*nonsparse_entries)
            nonsparse_values = list(sorted_entries[0])
            nonsparse_indices = list(sorted_entries[1:])

        # Convert indices and values to numpy array and check dimensionality
        for i, ind in enumerate(nonsparse_indices):
            assert len(ind) == len(nonsparse_values)
            nonsparse_indices[i] = np.array(ind, dtype='int')
            self.nonsparse_indices = nonsparse_indices
        self.nonsparse_values = np.array(nonsparse_values)

        # Setting dtype will alter self.nonsparse_values
        dtype = dtype or self.nonsparse_values.dtype
        self.dtype = dtype

        # Build lookup for quick indexing along the main_axis
        #   lookup defines first position of that element
        self.lookup = np.searchsorted(nonsparse_indices[self.main_axis],
                                      range(self.shape[self.main_axis]+1))

    @property
    def shape(self):
        if len(self.nonsparse_values):
            return tuple([max(inds)+1 for inds in self.nonsparse_indices])
        else:
            return tuple([0]*self.ndims)

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

            inds = range(*self.lookup[keys:keys+2])

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

        if not isinstance(keys, (int, tuple, list, slice, np.ndarray)):
            raise ValueError

        if isinstance(keys, slice):
            start, stop, step = keys.indices(len(self))
            keys = range(start, stop, step)

        assert isinstance(keys, int) or isinstance(keys[0], int), 'Indexing is only allowed along the main axis ({})'.format(self.main_axis)

        indices, values = self._nonsparse_entries(keys)

        if isinstance(keys, int):
            #Drop singleton dimension
            indices.pop(self.main_axis)

        tensor = self.__class__(dtype=self.dtype,
                                nonsparse_indices=indices, nonsparse_values=values,
                                main_axis=self.main_axis,
                                default_value=self.default_value)

        return tensor

    def __repr__(self):
        return "%s(dtype='%s', nonsparse_indices=%r, nonsparse_values=%r, main_axis=%r, default_value=%r)" % (
                self.__class__.__name__, self.dtype,
                [list(ind) for ind in self.nonsparse_indices],
                list(self.nonsparse_values), self.main_axis, self.default_value)

    def __str__(self):
        return "%s(dtype='%s', shape=%s, default_value=%s)" % (
                self.__class__.__name__, self.dtype, self.shape, self.default_value)

    def __eq__(self, other):
        ''' Returns true if the sparse matrix can be expressed as other (by for-
        cing it into the same shape).

        If shapes cannot match, raises
        '''
        if isinstance(other, SparseTensor):
            other = other.as_array()
            shape = [max(s,o) for s,o in zip(self.shape, other.shape)]
        else:
            other = np.array(other)
            shape = other.shape

        return self.as_array(shape) == other

    def __ne__(self, other):
        return np.invert(self == other)

    # Export and import functionality
    @classmethod
    def from_array(cls, arr, dtype=None, main_axis=0, default_value=0):
        ''' Turns a regular array or array-like into a SparseTensor

        # Arguments:
            arr (array-like): The array to convert into a SparseTensor
            dtype (str/np.dtype): The datatype to use. If none is provided then
                `np.array(arr).dtype` will be used
            default_value (of same dtype): The nonsparse value to filter out

        # Returns:
            tensor (selfDataTensor): S.t. `tensor.as_array(arr.shape) == arr`

        '''

        arr = np.array(arr)

        nonsparse_indices = list(np.where(arr != default_value))
        nonsparse_values = arr[nonsparse_indices]

        # Assume_sorted is main_axis=0 because of np.where
        assume_sorted = main_axis==0

        return cls(dtype=arr.dtype, nonsparse_indices=nonsparse_indices,
                   nonsparse_values=nonsparse_values, main_axis=0,
                   assume_sorted=assume_sorted)


    def as_array(self, shape=None):
        '''Returns the SparseTensor as a nonsparse np.array

        # Arguments:
            shape (tuple/list): desired output shape, if None, the minimal shape
                will be used. None values can also be used for individual dimensions
                wihin the shape tuple/list.
                Note that shape should be at least as big as `self.shape`.

        # Returns:
            out (np.array): nonsparse array of self.dtype

        '''

        if not shape:
            shape = [None] * self.ndims

        # Overwrite None values
        shape = [true_s if s==None else s for s, true_s in zip(shape, self.shape)]

        assert np.all([s >=true_s for s, true_s in zip(shape, self.shape)])

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
            return dict(nonsparse_indices=[i.tolist() for i in self.nonsparse_indices],
                        nonsparse_values=self.nonsparse_values.tolist(),
                        default_value=self.default_value, dtype=str(self.dtype),
                        main_axis=self.main_axis)
        else:
            return dict(nonsparse_indices=self.nonsparse_indices,
                        nonsparse_values=self.nonsparse_values,
                        default_value=self.default_value, dtype=str(self.dtype),
                        main_axis=self.main_axis)

    @classmethod
    def from_config(cls, config):
        ''' Returns a SparseTensor based on the `config` dict
        '''
        return cls(nonsparse_indices=config['nonsparse_indices'],
                    nonsparse_values=config['nonsparse_values'],
                    default_value=config['default_value'], dtype=config['dtype'],
                    main_axis=config['main_axis'], assume_sorted=True)


def unit_tests(seed=None):

    np.random.seed(seed)

    arr = np.random.randint(3, size=(50,30,5,8))
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
    assert np.all(sparse.as_array(full_shape) == sparse)

    print('Testing: Equality functions return `True` for all entries when comparing sparse with original array')
    assert np.all(arr == sparse.as_array(full_shape))
    assert np.all(sparse.as_array(full_shape) == arr)

    print('Testing: Equality functions should return same boolean array as numpy')
    assert np.all((arr[0] == 0) == (sparse[0] == 0))
    assert np.all((arr[0] == arr[3]) == (sparse[0] == sparse[3]))

    print('Testing: Inequality functions return `False` for all entries when comparing sparse with sparse')
    assert not np.all(sparse != sparse.as_array(full_shape))
    assert not np.all(sparse.as_array(full_shape) != sparse)

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

    print('All unit tests passed!')

class TensorList(object):
    ''' Helperclass to cluster tensors together, acts as a single list by propageting
    calls and slicing trough it's members.

    # Arguments:
        tensors (list of iterables): Should have the same length

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


class SparseTensorList(TensorList):
    ''' Same as `Tensorlist`, but enforces all `tensors` to be `SparseTensors`

    # Arguments:
        tensors (list of iterables): as in `TensorList`
        sparse_tensor_params: keyword arguments dict used when initialising
            the `SparseTensor`s

    '''
    def __init__(self, tensors, sparse_tensor_params={}):
        for i, tensor in enumerate(tensors):
            if not isinstance(tensor, SparseTensor):
                tensors[i] = SparseTensor.from_array(tensor, **sparse_tensor_params)

        super(SparseTensorList, self).__init__(tensors)

class SparseIterator(object):
    ''' Iterates over a dataset.

    # Arguments:
        data (iterable): data to iterate trough. The iterator will use list-indexing
            to retrieve batches from the data, so the data should support list-indexing
        batch_size (int): Number of datapoints yielded per batch
        shuffle (bool): Wether to loop trough the data randomly or not
        epochs (int/None): maximum number of epochs after which a `StopIteration`
            is raised (None for infinite generator)

    # Yields
        batch (iterable): a slice of `data` of length `batch_size` (except
            possibly on last batch of epoch)

    # Example
        using `keras.models.model`
        >>> model.fit_generator(SparseIterator(np.array(zip(data, labels)))

    # Note
        designed for use with keras `model.fit_generator`
    '''
    def __init__(self, data, batch_size=1, epochs=None, shuffle=True):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

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

        return self.data[use_inds]

    def reset(self):
        ''' Resets the counters of the iterator
        '''
        self.i = 0
        self.epoch = 0
        self.indices = range(len(self.data))


def example(simple=True):
    if simple:
        atoms = np.array(range(10))
        bonds = np.array(range(10, 20))
        edges = np.array(range(20, 30))

        labels = np.array(range(0, -10, -1))
        batch_size = 3
    else:
        atoms = np.random.randint(3, size=(50,30,62))
        bonds = np.random.randint(3, size=(50,30,5,5))
        edges = np.random.randint(3, size=(50,30,5,8))

        labels = np.random.rand(50)
        batch_size = 25

    mols = SparseTensorList([atoms, bonds, edges])
    data = TensorList([mols, labels])
    it = SparseIterator(data, epochs=100, batch_size=batch_size)

    for x, y in it:
        print('X:', x)
        print('Y', y)

if __name__ == '__main__':
    unit_tests()
    example()