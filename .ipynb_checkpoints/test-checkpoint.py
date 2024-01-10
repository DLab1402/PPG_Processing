from dataLoad import datLoDe
from vmd_feature import vmdFea, VMD
import torch
import scipy.sparse as sp
import numpy as np
import matplotlib
from model_1 import Net
import json

data = datLoDe("output_file.json").throwDatPack()
data = np.array(data)
# H = vmdFea(data,60).single_process(data[0,:],0.5,5,ReS = 130, selec_method = 'SSS')
fea = vmdFea(data,60)
H = fea.whole_pack_process(0.5,5,Re = 130, se_me = 'SSS')

print(type(H))
with open("feature.json", 'w') as json_file:
    # Use json.dump with additional parameters to handle complex data structures
    json.dump(H, json_file)