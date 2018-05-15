import stats as st, numpy as np

# y = b0 + b1 * x
class SimpleLinearRegression(object):
    b0 = None
    b1 = None
    
    def __init__(self):
        b0 = 0
        b1 = 0
    
    def fit(self,X,y):
        n = len(X)
        # _x e _y serao as medias de x e y. 
        _x = st.mean(X)
        _y = st.mean(y)
        
        self.__class__.b1 = np.sum((X - _x)*(y - _y)) / np.sum((X - _x) ** 2)
        self.__class__.b0 = _y - (self.__class__.b1 * _x)
    
    def predict(self,X):
        y_pred = self.__class__.b0 + self.__class__.b1*X
        return y_pred