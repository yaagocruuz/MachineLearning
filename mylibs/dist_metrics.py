import numpy as np

def minkowski_distance(X, X_row, p):
    X_ = np.abs(X - X_row) ** p
    return np.sum(X_, axis=1) ** (1/p)

def euclidean_distance(X, X_row):
    return minkowski_distance(X, X_row, 2)

def manhattan_distance(X, X_row):
    return minkowski_distance(X, X_row, 1)

def chebyshev_distance(X, X_row):
    return np.max(X - X_row)