import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay

class train:
    def __init(self,model,data,):
        self.model = model
        self.data = data 

    def train_epoch(model, device, loss_fn, data, optimizer, performance, display = 0):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            acc, _, _ = performance(outputs,labels,display)
            train_loss += loss.cpu().detach().numpy()
            train_acc += acc
        train_loss = train_loss / len(data)
        train_acc = train_acc / len(data)
        return train_loss, train_acc

    def val_epoch(model, device, loss_fn, data, performance, display = 0):
        val_loss = 0.0
        model.eval()
        total_labels = []
        total_ouputs = []
        with torch.no_grad():
            for inputs, labels in data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs,labels)
                val_loss += loss.cpu().detach().numpy()
                total_labels.append(labels[0])
                total_ouputs.append(outputs[0])
            val_loss = val_loss / len(data)
        total_labels = torch.stack(total_labels)
        total_ouputs = torch.stack(total_ouputs)
        val_acc, _, _ = performance(total_ouputs,total_labels, display)
        return val_loss, val_acc
    
    def training(self, performance, criterion = nn.CrossEntropyLoss(), lrate = 0.001, epoch = 100, batch = 64, visualize = 1):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu') 
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],'model':[]} 
        
        for fold, (train_idx,val_idx) in enumerate(splits.split(train_data)):
            fold_train_loss = []
            fold_val_loss = []
            fold_train_acc = []
            fold_val_acc = []
            fmodel = self.model.to(device)
            optimizer = optim.Adam(fmodel.parameters(), lr=lrate)
            if (visualize == 1):
                print("Fold :",fold+1)
                print("Train and validate dataset are overlap: ",any(x in val_idx for x in train_idx))
            for epoch in range(num_epochs):
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                #loss calculation
                train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler= train_idx)
                val_dataloader = torch.utils.data.DataLoader(train_data,batch_size=1, sampler= val_idx)
                train_loss, train_acc = train_epoch(fmodel, device, criterion, train_dataloader, optimizer, performance)
                val_loss, val_acc = val_epoch(fmodel, device, criterion, val_dataloader, performance)
                fold_train_loss.append(train_loss)
                fold_val_loss.append(val_loss)
                fold_train_acc.append(train_acc)
                fold_val_acc.append(val_acc)
                if (visualize == 1):
                    print('Iteration: {}/{}. loss_train: {}. loss_val: {}. train_acc: {}. val_acc: {}'.format(epoch, num_epochs,train_loss, val_loss, train_acc, val_acc))
            
            history['model'].append(fmodel)
            history['train_loss'].append(fold_train_loss[-1])
            history['val_loss'].append(fold_val_loss[-1])
            history['train_acc'].append(fold_train_acc[-1])
            history['val_acc'].append(fold_val_acc[-1])
            
            if (visualize == 1):
                plt.plot(fold_train_loss)
                plt.plot(fold_val_loss)
                plt.ylim([0, 1])
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend(['train_loss','val_loss'])
                plt.grid()
                plt.show()
                plt.plot(fold_train_acc)
                plt.plot(fold_val_acc)
                plt.ylim([0, 1])
                plt.xlabel("epoch")
                plt.ylabel("accuaracy")
                plt.legend(['train_acc','val_acc'])
                plt.grid()
                plt.show()
            
            return history
            
            