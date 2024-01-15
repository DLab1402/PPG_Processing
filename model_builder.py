import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay

def suggest_performance(outputs,labels, display = 0):
    y_true = []
    y_pred = []
    for idx in range(len(outputs)):
        if (outputs[idx][0]>outputs[idx][1]):
            y_pred.append([0])
        if (outputs[idx][1]>outputs[idx][0]):
            y_pred.append([1])
        if (labels[idx][0] == 1) &(labels[idx][1] == 0):
            y_true.append([0])
        if (labels[idx][0] == 0) &(labels[idx][1] == 1):
            y_true.append([1])
            # error.append(loss.detach().numpy())
    cm = confusion_matrix(y_true, y_pred)
    cm_d = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [True, False])
    # Compute the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = 0
    specificity = 0
    # Compute the specificity and sensitivity
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / (tn + fp)
#     sensitivity = tp/(tp+fn)
#     # Compute the ROC curve and area under the curve (AUC)
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#     roc_auc = auc(fpr, tpr)
    
#     if (display == 1):
#         cm_d.plot()
#         plt.show()
#         print('==============================Depticon==============================')
#         print('Accuracy:', accuracy)
#         print('Specificity:', specificity)
#         print('Sensitivity:', sensitivity)
#         print('ROC AUC:', roc_auc)
#         print('==============================Depticon==============================')
    return accuracy, sensitivity, specificity
    

class model_build:
    history = []
    def __init__(self,model,data,fold = 5, ratio = 0.2, shuff=True, r_state=42, criterion = nn.CrossEntropyLoss()):
        self.criterion = criterion
        self.model = model
        N = len(data)
        num_train = math.floor((1-ratio)*N)
        num_test = N - num_train
        train_data, test_data = random_split(data, [num_train, num_test])
        splits=KFold(n_splits=fold,shuffle=shuff,random_state=r_state)
        self.fold = splits.split(train_data)
        self.train_data = train_data
        self.test_data = test_data

    def train_epoch(self,model, device, loss_fn, data, optimizer, performance, display = 0):
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

    def val_epoch(self,model, device, loss_fn, data, performance, display = 0):
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
    
    def training(self, num_epochs = 10, performance = suggest_performance, lrate = 0.001, batch = 64, visualize = 1):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        
        for fold, (train_idx,val_idx) in enumerate(self.fold):
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
                train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, sampler= train_idx)
                val_dataloader = torch.utils.data.DataLoader(self.train_data,batch_size=1, sampler= val_idx)
                train_loss, train_acc = self.train_epoch(fmodel, device, self.criterion, train_dataloader, optimizer, performance)
                val_loss, val_acc = self.val_epoch(fmodel, device, self.criterion, val_dataloader, performance)
                fold_train_loss.append(train_loss)
                fold_val_loss.append(val_loss)
                fold_train_acc.append(train_acc)
                fold_val_acc.append(val_acc)
                if (visualize == 1):
                    print('Iteration: {}/{}. loss_train: {}. loss_val: {}. train_acc: {}. val_acc: {}'.format(epoch, num_epochs,train_loss, val_loss, train_acc, val_acc))
            
            tem = {'model': fmodel, 'train_loss': fold_train_loss, 'val_loss': fold_val_loss, 'train_acc': fold_train_acc,'val_acc':fold_val_acc}
            self.history.append(tem)
            
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
    
    def testing(self, performance = suggest_performance, visualize = 1):
        test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=1, shuffle=True)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        for fold_model in self.history:
            loss, acc = self.val_epoch(fold_model["model"], device, self.criterion, test_dataloader, performance)
            
        return loss, acc