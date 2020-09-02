import torch
from SocialLSTM import *
import sys

def showParams(path):
    model = torch.load(path)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)   
    for name, param in model.named_parameters():    
        if param.requires_grad:
            print(name, param.data)


def showError(path):
    


if __name__ == "__main__":
    argv = sys.argv
    if argv[1] == "print":
        showParams(argv[2])
    elif argv[1] == "print":
        showError(argv[2])
     