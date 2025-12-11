import pandas as pd 
import numpy as np

class LabelEncode:
    def __init__(self):
        super().__init__()
    
    def fit_transform(self,x:pd.DataFrame):
        '''
        Docstring for fit_transform
        
        :param x: Enter the data frame 
        :type x: pd.DataFrame
        :param is_label_dict_needed: If the Dictonary of Label is needed then make this True else Default is False
        '''
        vocab = []
        for i in x.columns:
            if x[i].dtype == 'object':
                vocab.append(np.array(x[i]))
        
        vocab_array = np.concatenate(vocab,axis=0)
        is_present = set()
        labels_dict = {}
        labels  = []
        for i in range(len(vocab_array)):
            if vocab_array[i] not  in is_present:
                is_present.add(vocab_array[i])
                labels_dict[vocab_array[i]] = i
                labels.append(i)
            else:
                pass
        
        return labels_dict