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
from PIL import Image
import torchvision
import os
import math

model = Net()

model.load_state_dict(torch.load("Train_result/model_1.pt"))
model.eval()

transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class auto_process:
    def __init__(self,data,fs):
        self.data = data
        self.fs = fs
        
    def __seg_eval(self,data):
        tem = vmdFea(data,self.fs).single_process(self.data,0.5,5,ReS = 130, selec_method = 'SSS')
        normalized_image = (tem - np.min(tem)) * (255.0 / (np.max(tem) - np.min(tem)))
        image = Image.fromarray(normalized_image.astype('uint8'))
        image = image.resize((512, 128))
        img = image.convert('RGB')
        img = transform(img)
        x = model.forward(torch.unsqueeze(img, 0))
        if x[0][0]>x[0][1]:
            return True
        else:
            return False

    def auto_seg(self, n, ratio = 0.1, pack = False, visualize = False):
        x = math.ceil(n*ratio)
        ite = n-2*x
        num = math.ceil(len(self.data)/ite)

#         segIn1 = np.array(np.arange(num)*ite).astype(np.int64)
#         segIn2 = np.array(np.arange(num)*ite+x).astype(np.int64)
#         segIn3 = np.array(np.arange(1,num+1)*ite+x-1).astype(np.int64)
#         segIn4 = np.array(np.arange(1,num+1)*ite+2*x-1).astype(np.int64)

        segIn1 = [i*ite for i in range(num)]
        segIn2 = [i*ite+x for i in range(num)]
        segIn3 = [(i+1)*ite+x-1 for i in range(num)]
        segIn4 = [(i+1)*ite+2*x-1 for i in range(num)]
        
#         print(type(segIn1))
#         print(type(segIn2))
#         print(type(segIn3))
#         print(type(segIn4))

        take = []
        reject = []
        
        if visualize == True:
            index = np.zeros(len(self.data))
            
        for i in range(num):
            e = self.__seg_eval(self.data[segIn1[i]:segIn4[i]])
            if e == True:
                take.append([segIn2[i],segIn3[i]])
                if visualize == True:
                    index[segIn2[i]:segIn3[i]] = 1            
            else:
                reject.append([segIn2[i],segIn3[i]])
        
        if visualize == True:
            plt.plot(self.data)
            plt.plot(index*(np.max(data)-np.min(data)))
            plt.show
            plt.savefig("result.png")
        
        if pack == True:
            pack = []
            for i in range(len(take)):
                pack.append(self.data[take[i][0]:take[i][1]])
            return pack
        else:
            return np.array(take), np.array(reject)
                
#Test script

# data = datLoDe("output_file.json").throwDatPack()
# data = np.array(data)
# x = auto_process(data[0][0:420],60)
# p = x.auto_seg(100, pack = True, visualize = True)
# print(len(p))