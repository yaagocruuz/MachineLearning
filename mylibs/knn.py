import numpy as np
import dist_metrics as dm

class KNNClassifier(object):
    X = None
    y = None
    
    def __init__(self, k=5, metric='minkowski', p=2):
        self.k = k
        self.metric = metric
        self.p = p
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    
    def predict(self, X_test):
        if(self.metric == 'minkowski'):
            distances = dm.minkowski_distance(self.X, X_test, self.p)
        elif(self.metric == 'euclidean'):
            distances = dm.euclidean_distance(self.X, X_test)
        elif(self.metric == 'manhattan'):
            distances = dm.manhattan_distance(self.X, X_test)
        elif(self.metric == 'chebyshev'):
            distances = dm.chebyshev_distance(self.X, X_test)
        
        idx_sort = np.argsort(distances)
        
        output_values = self.y[idx_sort[1:self.k+1]]
        
        counts = np.unique(output_values,return_counts=True)
        idx_max = np.argmax(counts[1])
        return counts[0][idx_max]
    
    
class KNNRegressor(object):
    X = None
    y = None
    
    def __init__(self, k=5, metric="minkowski", p=2):
        self.k = k
        self.metric = metric
        self.p = p
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X_test):
        if(self.metric == 'minkowski'):
            distances = dm.minkowski_distance(self.X, X_test, self.p)
        elif(self.metric == 'euclidean'):
            distances = dm.euclidean_distance(self.X, X_test)
        elif(self.metric == 'manhattan'):
            distances = dm.manhattan_distance(self.X, X_test)
        elif(self.metric == 'chebyshev'):
            distances = dm.chebyshev_distance(self.X, X_test)
           
        idx_sort = np.argsort(distances)
        output_values = self.y[idx_sort[1:self.k+1]]
        
        return np.sum(output_values) / output_values.shape[0]
    
    
    
    
    