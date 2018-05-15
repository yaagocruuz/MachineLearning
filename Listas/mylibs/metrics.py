from math import sqrt
import numpy as np

def mse(y_true, y_pred):
    n_samples = len(y_true)
    _mse = 0
    i = 0
    for v in y_true:
        _mse += ((v - y_pred[i]) ** 2)
        i += 1
    _mse = (1/n_samples) * _mse
    return _mse

def rmse(y_true, y_pred):
    _rmse = sqrt(mse(y_true, y_pred))
    return _rmse

def mae(y_true, y_pred):
    n_samples = len(y_true)
    _mae = 0
    i = 0
    for v in y_true:
        _mae += np.abs((v - y_pred[i]))
        i += 1
    _mae = (1/n_samples) * _mae
    return _mae