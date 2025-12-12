import numpy as np 
import pandas as pd
'''

w.shape = (12,4) , x1.shape = (12,4)

w.T * x1 

'''

class Perceptron:
    def __init__(self,num_iter=100,lr_rate=0.001):
        super().__init__()
        self.epoch = num_iter
        self.lr_rate = lr_rate
    
    def train_test_split(self,X,y,test_size=0.2):
        '''
        Docstring for train_test_split
        : It will return x_train,x_test,y_train,y_test
        
        :param x: Dataset Featutres 
        :param y: Label (Target)
        :param test_size: Test Size (Default is 20%)
        '''
        xn = X.shape[0]
        yn = y.shape[0]
        x_test_size = int(X.shape[0] * test_size)
        y_test_size = int(y.shape[0] * test_size)
        
        x_train = X[:xn - x_test_size]
        x_test = X[xn - x_test_size: xn]
        y_train = y[:yn - y_test_size]
        y_test = y[yn - y_test_size:yn ]
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # print(f"y_train shape : {y_train.shape}\nx_train shape: {x_train.shape}\ny_test shape: {y_test.shape}\nx_test shape: {x_test.shape}")
        return x_train,x_test,y_train,y_test
    
    def sigmoid(self,x):
        x = 1 / (1 + np.exp(-x))
        return x
    
    def feed_forward(self,x):
        z = np.dot(x,self.w) + self.bias
        return z
    
    def fit(self,x,y):
        n_input,n_features = x.shape
        self.w = np.random.randn(n_features) / np.sqrt(n_features)
        self.bias = 0.001
        losses = []
        for _ in range(self.epoch):
            loss = 0
            for xi,target in zip(x,y):
                y_cap = self.feed_forward(xi)
                y_hat = 1 if y_cap > 0 else 0
                update_ = self.lr_rate  * (target - y_hat)
                self.w += update_ * xi
                self.bias +=  update_
                loss += int(update_ != 0)
            losses.append(loss)
        return losses
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.bias
        return np.where(linear_output >= 0, 1, 0)

if __name__ ==  '__main__':
    model = Perceptron(num_iter=100,lr_rate=0.001)
