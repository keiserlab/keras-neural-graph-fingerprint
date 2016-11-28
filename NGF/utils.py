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