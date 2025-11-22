import numpy as np 


def leaky_relu(x:np.ndarray,alpha=0.01):
    
    '''
    * Provide alpha otherwise Default is 0.01
    
    * formula:    x if x > 0 else a*x
    
    * Range : (-inf , +inf)
    ''' 
    x = np.where(x > 0,x ,alpha*x)
    return x

