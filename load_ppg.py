import csv
import torch
import math
from sklearn.model_selection import KFold
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader

class data_call:    
    data = None
    def __init__(self, file_link):
        self.file_link = file_link        
        
    def load_data(self):
        file = open(self.file_link)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            tensor_data = torch.FloatTensor([float(value) for value in row])
            rows.append(tensor_data)
        self.data = torch.stack(rows)
        
    def data_setup(self, fold = 5, ratio = 0.2, shuffle=True, random_state=42):
        self.load_data()
        N = len(self.data)
        num_train = math.floor((1-ratio)*N)
        num_test = N - num_train
        train_data, test_data = random_split(self.data, [num_train, num_test])
        splits=KFold(n_splits=fold,shuffle=True,random_state=42)
        train_data = splits.split(train_data)
        return train_data, test_data
        
#test script

# file_link = "PPG_PO_train.csv" 
# a = data_call(file_link)
# train, test = a.data_setup()
# d = enumerate(train)
# print(len(d))


# print(type(torch.tensor(a.data)))