import tifffile
import numpy as np
import random
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import random
import torchvision.transforms as transforms
from collections import Counter
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# extracted subvolumes
data = tifffile.imread('berea.tif').astype(np.int8)
data = np.abs(data)
print(data.shape)
print(1-np.mean(data))
data.tofile("berea.raw")
flag = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            np.save("%d.npy"%(flag),data[0+93*i:128+93*i,0+93*j:128+93*j,0+93*k:128+93*k].flatten())
            flag +=1
# extracted porosity
flag = 0
pore = np.zeros((125*128,1))
for i in range(125):
    data = np.load("%d.npy"%(i)).reshape((128,128,128))
    for j in range(128):
        pore[flag] = 1-np.mean(data[j])
        flag +=1

    print(i)
np.save("pore.npy",pore)
p = np.load("pore.npy")
data = np.load("5.npy").reshape((128,128,128))
print(1-np.mean(data[0]))
print(p[128*5])






# # data_new = np.zeros((81,128,128,128))
# # pos = 0
# # i=8
# # # for i in tqdm(range(9)):
# # for j in tqdm(range(9)):
# #     for k in range(9):
# #         data_new[pos]=data[i*34:i*34+128,j*34:j*34+128,k*34:k*34+128]
# #         pos = pos+1
# #
# #
# # np.save("data_new_%d.npy"%(i),data_new.flatten())
# j = 8
# data = np.load("data_new_%d.npy"%(j)).reshape((-1,128,128))
#
# # print(Counter(data[0].flatten()))
# pore = np.zeros((len( data),1))
# for i in tqdm(range(len(data))):
#     pore[i] = Counter(data[i].flatten())[0.0]/(128*128)
#     # print(Counter(data[i].flatten())[0.0])
# np.save("pore_%d.npy"%(j),pore)
# #
# # data = np.load("pore_8.npy")
# # print(data)
# # data= np.fromfile("MULTI_SAND.raw",dtype=np.int8).reshape((512,512,512))
# # data = np.abs(data)
# # data_new = np.zeros((3,128,128,128))
# # for i in range(3):
# #     data_new[i] = data[i*128:(i+1)*128,128:256,256:]
# #     print(i)
# # plt.imshow(data_new[0,0])
# # print(data_new[0,0])
# # plt.show()
# # np.save("data_beadpack_5.npy",data_new.flatten())
# #
# # ls = os.listdir('beadpack')
# # print(ls)
# # os.chdir("beadpack")
# # data_new = np.zeros((27,128,128,128))
# # i = 0
# # for data_name in ls:
# #     data = np.load(data_name).reshape((3,128,128,128))
# #     data_new[i:i+3] = data
# #     i += 3
# #     print(i)
# # np.save("data_beadpack.npy",data_new.flatten())
# # def p(img):
# #     data =img.flatten()
# #     cnt = 0
# #     for i in range((len(data))):
# #         if data[i] == 0 :cnt += 1
# #     return cnt/len(data)
#
#
# # data = np.load('berea/data_berea.npy').reshape((-1,128,128,128))
# # print(data.shape)
# # pore = np.zeros((27,128))
# # for i in range(27):
# #     for j in range(128):
# #         pore[i,j] = p(data[i,j])
# #         print(i)
# # np.savetxt("berea/pore_3d.npy",pore.flatten())