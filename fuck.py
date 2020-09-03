import torch
from torch import nn
loss = nn.MSELoss(reduction='none')
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
