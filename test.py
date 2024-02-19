import numpy as np
from vmd_feature import vmdFea
from PIL import Image
import matplotlib.pyplot as plt
import json
file_path = "G:/My Drive/REPORT/AI_in_medical/Thong_nhat_project/bai_bao/my_data.json"
# file_path = "G:/My Drive/PPG_Process/ppg_processing/output_file.json"
with open(file_path, 'r') as json_file:
    json_data = json.load(json_file)
fea = vmdFea([],125)

def save_imag(name,tem,file_path):
    normalized_image = (tem - np.min(tem)) * (255.0 / (np.max(tem) - np.min(tem)))
    image = Image.fromarray(normalized_image.astype('uint8'))
    image = image.resize((512, 128))
    image.save(file_path+"/"+name+".png")

plt.plot(json_data[1][0])
plt.plot(json_data[10][0])
plt.show()

for index,data in enumerate(json_data):
    H = fea.single_process(data[0], 0.3, 5,ReS = 100, t_cut=0.1, selec_method = "AAA")
    # if index == 1:
    #     plt.plot(data[0])
    #     plt.show()

    # if index == 10:
    #     plt.plot(data[0])
    #     plt.show()
    if data[1] == 0:
        file_path = "G:/My Drive/REPORT/AI_in_medical/Thong_nhat_project/bai_bao/0"
        save_imag(str(index)+"-0",H,file_path)
    if data[1] == 1:
        file_path = "G:/My Drive/REPORT/AI_in_medical/Thong_nhat_project/bai_bao/1"
        save_imag(str(index)+"-1",H,file_path)
    if data[1] == 2:
        file_path = "G:/My Drive/REPORT/AI_in_medical/Thong_nhat_project/bai_bao/2"
        save_imag(str(index)+"-2",H,file_path)