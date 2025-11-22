import numpy as np 



def sigmoid(x):
    '''
    formula : 1 / 1 + e^-x
    '''
    x = 1 / (1 + np.exp(-x))
    return x
