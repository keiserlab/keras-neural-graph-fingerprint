# Imported from https://github.com/GUR9000/KerasNeuralFingerprint/blob/master/KerasNeuralfingerprint/utils.py

from __future__ import print_function

import csv
import numpy as np
import itertools as it

def read_csv(filename, nrows, input_name, target_name):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in it.islice(reader, nrows):
            data[0].append(row[input_name])
            data[1].append(float(row[target_name]))
    return map(np.array, data)

def permute_data(data, labels=None, FixSeed=None, return_permutation=False, permutation = None):
    """Returns:
    data, labels (if both given) otherwise just data   , permutation [iff return_permutation==True]"""
    if FixSeed!=None:
        np.random.seed(int(FixSeed))
    s = np.shape(data)[0]
    if permutation is None:
        per = np.random.permutation(np.arange(s))
    else:
        per = permutation
    if type(data)==type([]):
        cpy = [data[i] for i in per]
    else:
        cpy = data[per]    #creates a copy! (fancy indexing)
    if labels is not None:
        if type(labels)==type([]):
            cpyl = [labels[i] for i in per]
        else:
            cpyl = labels[per]
        if not return_permutation:
            return cpy, cpyl
        else:
            return cpy, cpyl, per
    if not return_permutation:
        return cpy
    else:
        return cpy,per

def load_delaney(file = 'data/delaney.csv', target_name = 'measured log solubility in mols per litre'):
    '''
    returns: data, labels
    '''

    _alldata = read_csv(file, 1128, input_name='smiles', target_name=target_name)
    assert len(_alldata[0])==len(_alldata[1])
    data, labels = permute_data(_alldata[0], _alldata[1], FixSeed=12345)
    assert len(data)==len(labels)
    return data, labels
        
def load_Karthikeyan_MeltingPoints(file = 'data/Melting_Points_(Karthikeyan).txt', target_name='MTP'):
    '''
    returns: data, labels
    '''
    _alldata = read_csv(file, 4451, input_name='SMILES', target_name=target_name)
    assert len(_alldata[0])==len(_alldata[1])
    data, labels = permute_data(_alldata[0], _alldata[1], FixSeed=12345)
    assert len(data)==len(labels)
    return data, labels