''' Utilities used within the NGF module
'''
from __future__ import print_function

import inspect
from itertools import cycle

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

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def zip_mixed(*mixed_iterables, **kwargs):
    ''' Zips a mix of iterables and non-iterables, non-iterables are repeated
    for each entry.

    # Arguments
        mixed_iterables (any type): unnamed arguments (just like `zip`)
        repeat_classes (list): named argument, which classes to repeat even though,
            they are in fact iterable

    '''

    repeat_classes = tuple(kwargs.get('repeat_classes', []))
    mixed_iterables = list(mixed_iterables)

    for i, item in enumerate(mixed_iterables):
        if not is_iterable(item):
            mixed_iterables[i] = cycle([item])

        if isinstance(item, repeat_classes):
            mixed_iterables[i] = cycle([item])

    return zip(*mixed_iterables)

def mol_dims_to_shapes(max_atoms, max_degree, num_atom_features, num_bond_features, num_molecules=None):
    ''' Helper function, returns shape for molecule tensors given dim sizes
    '''
    atoms_shape = (num_molecules, max_atoms, num_atom_features)
    bonds_shape = (num_molecules, max_atoms, max_degree, num_bond_features)
    edges_shape = (num_molecules, max_atoms, max_degree)
    return [atoms_shape, bonds_shape, edges_shape]

def mol_shapes_to_dims(mol_tensors=None, mol_shapes=None):
    ''' Helper function, returns dim sizes for molecule tensors given tensors or
    tensor shapes
    '''

    if not mol_shapes:
        mol_shapes = [t.shape for t in mol_tensors]

    num_molecules0, max_atoms0, num_atom_features = mol_shapes[0]
    num_molecules1, max_atoms1, max_degree1, num_bond_features = mol_shapes[1]
    num_molecules2, max_atoms2, max_degree2 = mol_shapes[2]

    num_molecules_vals = [num_molecules0, num_molecules1, num_molecules2]
    max_atoms_vals = [max_atoms0, max_atoms1, max_atoms2]
    max_degree_vals = [max_degree1, max_degree2]

    assert len(set(num_molecules_vals))==1, 'num_molecules does not match within tensors (found: {})'.format(num_molecules_vals)
    assert len(set(max_atoms_vals))==1, 'max_atoms does not match within tensors (found: {})'.format(max_atoms_vals)
    assert len(set(max_degree_vals))==1, 'max_degree does not match within tensors (found: {})'.format(max_degree_vals)

    return max_atoms1, max_degree1, num_atom_features, num_bond_features, num_molecules1