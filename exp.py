import torch

X = torch.tensor([[[1.,2.],[3.,4.],[5.,6.]],[[1.,2.],[3.,4.],[5.,6.]],[[1.,2.],[3.,4.],[5.,6.]],[[1.,2.],[3.,4.],[5.,6.]]])
print(X, X.shape)
X = X.reshape(X.shape[0]*X.shape[1],X.shape[2])
print(X, X.shape)
Y = X.clone()*3
