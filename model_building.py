from dataLoad import datLoDe
from vmd_feature import vmdFea, VMD
import torch
import scipy.sparse as sp
import numpy as np
import matplotlib
from model_1 import Net
import json
from ImageDataLoad import ImageDataset
from model_builder import model_build
import matplotlib.pyplot as plt
import torchvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# data = datLoDe("output_file.json").throwDatPack()
# data = np.array(data)
# # H = vmdFea(data,60).single_process(data[0,:],0.5,5,ReS = 130, selec_method = 'SSS')
# fea = vmdFea(datav,60)
# fea.whole_pack_process(0.5,5,Re = 130, se_me = "SSS" ,file_path = "Train_data")

root = "Train_data"
dataset = ImageDataset(root)

model = Net()

builder = model_build(model,dataset)
builder.training(num_epochs = 30,visualize = 0)

for i,his in enumerate(builder.history, start = 1):
    torch.save(his["model"].state_dict(), "Train_result/model_"+str(i)+".pt")
    
    plt.plot(his["train_loss"])
    plt.plot(his["val_loss"])
    plt.show()
    plt.savefig("Train_result/model_loss"+str(i)+".png")
    plt.clf()

    plt.plot(his["train_acc"])
    plt.plot(his["val_acc"])
    plt.show()
    plt.savefig("Train_result/model_acc"+str(i)+".png")
    plt.clf()

l, a = builder.testing()
print('Performance on test set: loss_val: {}; acc_val: {}'.format(l, a))