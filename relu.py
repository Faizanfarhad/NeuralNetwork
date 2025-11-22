import numpy as np 


def relu(x):
    '''
    * x : input 
    
    * formula: x if x > 0 else 0
    
    * Range : (0 , +inf)
    '''
    x = np.max(0,x)
    return x


