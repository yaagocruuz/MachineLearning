import numpy as np

def standardize(X):
    X_std = np.copy(x)
    n_cols = X.shape[1]
    for i in range(n_cols):
        X_std[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])
    return X_std

def normalize(X):
    X_norm = np.copy(X)
    n_cols = X.shape[1]
    for i in tange(n_cols):
        X_norm[:,1] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))
    return X_norm