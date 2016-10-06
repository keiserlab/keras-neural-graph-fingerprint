# Imported from https://github.com/GUR9000/KerasNeuralFingerprint/blob/master/KerasNeuralfingerprint/utils.py

from __future__ import print_function

import csv
import numpy as np
import itertools as it

#from mol_graph import graph_from_smiles_tuple, degrees




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
    
    
def load_Karthikeyan_MeltingPoints(file = 'ata/Melting_Points_(Karthikeyan).txt', target_name='MTP'):
    '''
    returns: data, labels
    '''
    _alldata = read_csv(file, 4451, input_name='SMILES', target_name=target_name)
    assert len(_alldata[0])==len(_alldata[1])
    data, labels = permute_data(_alldata[0], _alldata[1], FixSeed=12345)
    assert len(data)==len(labels)
    return data, labels
    
    
    
    
    
def cross_validation_split(data, labels, crossval_split_index, crossval_total_num_splits, validation_data_ratio = 0.1):
    '''
    Manages cross-validation splits given fixed lists of data/labels
    
    
    <crossval_total_num_splits> directly affects the size of the test set ( it is <size of data-set>/crossval_total_num_splits)
    
    Returns:
    ----------
    
        traindata, valdata, testdata
    
    '''
    assert validation_data_ratio<1 and validation_data_ratio > 0
    assert crossval_split_index < crossval_total_num_splits
    
    N = len(data)
    n_test = int(N*1./crossval_total_num_splits)
    if crossval_split_index == crossval_total_num_splits - 1:
        n_test = N - crossval_split_index * n_test
    
    # <valid or train|[@crossval_split_index] test|valid or train>
    
    start_test = crossval_split_index * n_test
    end_test = crossval_split_index * n_test + n_test
    testdata = (data[start_test: end_test], labels[start_test: end_test])
    
    rest_data   = np.concatenate((data[:start_test],data[end_test:]), axis=0)
    rest_labels = np.concatenate((labels[:start_test],labels[end_test:]), axis=0)
    
    n_valid   = int(N * validation_data_ratio)
    valdata   = (rest_data[: n_valid], rest_labels[: n_valid])
    traindata = (rest_data[n_valid: ], rest_labels[n_valid: ])
    
    return traindata, valdata, testdata





def array_rep_from_smiles(smiles):
    """extract features from molgraph"""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep





def filter_data(data_loading_function, data_cache_name = 'default_data_cache/'):
    """
    loads data using <data_loading_function> (e.g. load_Karthikeyan_MeltingPoints()) and filters out all invalid SMILES.
    Saves the processed data on disk (name is specified by <data_cache_name>) and will re-load this file 
    the next time filter_data() is called if the same <data_cache_name> is provided
    
    Inputs:
    ---------
    
        data_loading_function:
        
            a function returning two lists: a list of smiles(input data) and a list of labels/regression targets
        
        
        data_cache_name:
        
            string describing the location for storing the filtered data on disk. 
            
            Set to None in order to disable this.
    """
    try: #try to load cached files
        if data_cache_name is not None:
            data   = np.load(data_cache_name+'_data.npy')        
            labels = np.load(data_cache_name+'_labels.npy')
    except:
        data_, labels_ = data_loading_function()# e.g. load_Karthikeyan_MeltingPoints()
        data, labels = [ ],[]
        ok, banned = 0,0
        for i in range(len(data_)):
            try:
                array_rep_from_smiles(data_[i:i+1])
                data.append(data_[i])
                labels.append(labels_[i])
                ok +=1
            except:
                banned +=1
        if data_cache_name is not None:
            print('removed', banned, 'and kept', ok,'samples')
        data = np.array(data)
        labels = np.array(labels)
        
        if data_cache_name is not None:
            np.save(data_cache_name+'_data.npy', data)
            np.save(data_cache_name+'_labels.npy', labels)
    return data, labels
        


