import numpy as np
from sklearn.metrics import classification_report
import cv2 as cv 

class LogisticRegression:
    def __init__(self,num_iter=1000, lr_rate=0.001):
        super().__init__()
        self.num_iter = num_iter
        self.lr_rate = lr_rate 
        self.weights = None 
        self.bias = None
        
    def sigmoid(self,x):
        x = 1 / (1 + np.exp(-x))
        # x = max(0,x)
        return x
    
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
        print(xn , yn , x_test_size , y_test_size)
        x_train = X[:xn - x_test_size]
        x_test = X[xn - x_test_size: xn]
        y_train = y[:yn - y_test_size]
        y_test = y[yn - y_test_size:yn ]
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        print(f"y_train shape : {y_train.shape}\nx_train shape: {x_train.shape}\ny_test shape: {y_test.shape}\nx_test shape: {x_test.shape}")
        return x_train,x_test,y_train,y_test

    def feed_forward(self,x):
        z = np.dot(x,self.w) + self.bias
        z = self.sigmoid(z)
        return z
    

    def fit(self,x:np.ndarray,y:np.ndarray,need_weights=False):
        '''
        Docstring for fit:
        :This Logistic Regression Mainly Focused on Image Classification
        
        :param x: features 
        :param y: labels
        :param need_weights: True if the weights are needed Else False
        '''
        
        self.w = np.random.randn(x.shape[1]) / np.sqrt(x.shape[1]/2)
        self.bias = 0.001
        
        for _ in range(self.num_iter):
            z = self.feed_forward(x)
            dw = self.compute_dw(x,z,y)
            db = self.compute_db(z,y)
            
            self.w  -= (self.lr_rate * dw)
            self.bias -= (self.lr_rate * db)
            
            if _ % 100 == 0: 
                binary_crossentropy = self.binary_crossentropy(z,y)
                print(f"After {_} Epochs BCE : {binary_crossentropy}")
        if need_weights:
            return self.w,self.bias 
        

    def compute_dw(self,x:np.ndarray,y_pred,y_true):
        n = y_pred.shape[0]
        u = y_pred - y_true
        dw = np.dot(x.T , u)
        dw = (1/n) *dw
        return dw 

    def compute_db(self,y_pred,y):
        n = y_pred.shape[0]
        u = y_pred - y
        db =  (1 / n) *np.sum(u)
        return db

    def evaluate(self,x_test:np.ndarray,y_test):
        '''
        :After evauate Call "evaluate_report" for getting model report
        
        :type x_test: np.ndarray
        '''
        y_pred  = self.feed_forward(x_test)
        preds = [1 if i > 0.5 else 0 for i in y_pred]
        target_names = ["Cat", "Dog"]
        self.report = classification_report(y_pred=preds,y_true=y_test,target_names=target_names)
        
        return self.report
    
    def predict(self,x,w,b):
        y_pred  = self.fortrained_feed_forward(x,w,b)
        preds = [1 if i > 0.5 else 0 for i in y_pred]
        return preds
        
    def fortrained_feed_forward(self,x,w,b):
        z = np.dot(x,w) + b
        z = self.sigmoid(z)
        return z

    def binary_crossentropy(self,y_pred:np.ndarray,y_true:np.ndarray):
        '''
        Docstring for loss
        
        :param y_pred: Predictions of the models 
        :param y:  y_true 
        '''
        epsilon = 1e-5
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        bce = -np.mean(y1 + y2)
        return bce

if __name__ == '__main__':
    model = LogisticRegression(num_iter=1000,lr_rate=0.001)
    
