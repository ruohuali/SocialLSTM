import torch
import numpy as np
import pickle    

with open("x_all.p", 'rb') as f:
    file = pickle.load(f)

print(file.shape)
x = file[1,2,:]
print(x)
print(x.shape)