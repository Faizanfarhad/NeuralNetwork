import numpy as np
from sklearn.metrics import classification_report

class LogisticRegression:
    def __init__(self,num_iter=1000, lr_rate=0.001, batch_size=256):
        super().__init__()
        '''
        :param num_iter: Number of iteration(epoc) 
        :param lr_rate: Learning Rate
        :param batch_size : Batchsize Default is 256
        '''
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.lr_rate = lr_rate 
        self.weights = None 
        self.bias = None
        
    def sigmoid(self,x):
        x = 1 / (1 + np.exp(-x))
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

    def mini_batch(self,x,y,w,batch_size):
        x_mini_batches = []
        y_mini_batches = []
        w_mini_batches = []
        n_minibatches = x.shape[0] // batch_size
        
        for i in range(n_minibatches):
            start = i * batch_size
            end = min(start + batch_size,x.shape[0])
            x_mini_batch = x[start:end]
            y_mini_batch = y[start:end]
            w_mini_batch = w[start:end]
            x_mini_batches.append(x_mini_batch)
            y_mini_batches.append(y_mini_batch)
            w_mini_batches.append(w_mini_batch)
        return x_mini_batches,y_mini_batches,w_mini_batches

    
    def feed_forward(self,x,w):
        x = np.array(x)
        w = np.array(w)
        z = np.dot(x,w.T) + self.bias
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
        
        self.w = np.random.randn(x.shape[0],x.shape[1]) / np.sqrt(x.shape[0]/2)
        self.bias = 0.001
        
        for _ in range(self.num_iter):
            x_mini_batch , y_mini_batch,w = self.mini_batch(x,y,self.w,self.batch_size)
            for i in range(len(x_mini_batch)):
                z = self.feed_forward(x_mini_batch[i],w[i])
                dw = self.compute_dw(x_mini_batch[i],z,y_mini_batch[i])
                db = self.compute_db(z,y_mini_batch[i])
                
                w[i] -= (self.lr_rate * dw.T)
                self.bias -= (self.lr_rate * db)

                binary_crossentropy = self.binary_crossentropy(z,y_mini_batch[i])
                print(f"After {i} Batch BCE : {binary_crossentropy}")
        if need_weights:
            w_stack = np.concatenate(w,axis=0)
            return w_stack,self.bias 
        

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

    def evaluate(self,x_test:np.ndarray,w,y_test):
        '''
        :After evauate Call "evaluate_report" for getting model report
        
        :type x_test: np.ndarray
        '''
        y_pred  = self.fortrained_feed_forward(x_test,w,self.bias)
        preds = [1 if i > 0.5 else 0 for i in y_pred]
        target_names = ["Cat", "Dog"]
        self.report = classification_report(y_pred=preds,y_true=y_test[:y_pred.shape[0]],target_names=target_names)
        
        return self.report
    
    def predict(self,x,w,b):
        y_pred  = self.fortrained_feed_forward(x,w,b)
        preds = [1 if i > 0.5 else 0 for i in y_pred]
        return preds
        
    def fortrained_feed_forward(self,x,w,b):
        '''
        Docstring for fortrained_feed_forward
        
        :param x: x_test 
        :param w: Trained Weights
        :param b: Trained Bias
        '''
        prediction = []
        y = np.zeros(x.shape[0])
        x_mini_batch , y_mini_batch,w_mini_batch= self.mini_batch(x,y,w,self.batch_size)
        for i in range(len(x_mini_batch)):
            z = np.dot(x[i],w_mini_batch[i].T) + b
            z = self.sigmoid(z)
            prediction.append(z)
        prediction = np.concatenate(prediction,axis=0)
        print(prediction.shape)
        return prediction

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
    model = LogisticRegression(num_iter=2,lr_rate=0.001)
    