from dataLoad import datLoDe
from vmd_feature import vmdFea, VMD
import torch
import scipy.sparse as sp
import numpy as np
import matplotlib
from model_1 import Net

data = datLoDe("G:/My Drive/REPORT/output_file.json").throwDatPack()
data = np.array(data)
H = vmdFea(data,60).sigle_process(data[0,:],0.5,5,ReS = 130, selec_method = 'SSS')
print(H.shape)
dense_tensor = torch.FloatTensor(np.array([[H[0:127,0:512]]]))
print(dense_tensor.shape)
model = Net()
y = model.forward(dense_tensor)
# y = torch.nn.Conv2d(1, 3, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False) 
# r = y(dense_tensor)