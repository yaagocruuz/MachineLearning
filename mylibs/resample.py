import numpy as np

def split_train_test(n_elem, perc_train, seed):
    ls = [i for i in range(n_elem)]
    np.random.seed(seed)
    np.random.shuffle(ls)
    aux = n_elem * perc_train
    return ls[:int(aux)], ls[int(aux):]

def split_k_fold(n_elem, n_splits=3, shuffle=True, seed=0):
    idx_test = []
    idx_train = []
    ls = [i for i in range(n_elem)]
    if shuffle == True:
        if seed != None:
            np.random.seed(seed)
        np.random.shuffle(ls)
    
    aux = n_elem / n_splits
    for i in range(n_splits):
        idx_test.append(ls[:int(aux)])
        idx_train.append(ls[int(aux):])
        ls = ls[int(aux):] + ls[:int(aux)]
        
    return idx_train, idx_test