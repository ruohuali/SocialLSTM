# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
from torch import nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import pdb 
import time
import matplotlib.pyplot as plt
from os import listdir, remove
from os.path import isfile, join
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from copy import deepcopy 
import pickle
import sys
import argparse

from models import *
from utils import *
from metrics import *


# %%
def train(T_obs, T_pred, files, model_type='v', model=None, name="model.pt", EPOCH=5):
    tic = time.time()
    print(f"type {model_type} totally training on {files}")    
    #params
    h_dim = 128

    if model == None:
        if model_type == 'v':
            print("instantiating model "+model_type)
            vl = VanillaLSTM(hidden_dim=h_dim, mediate_dim=32, output_dim=2)
        else:
            print("instantiating model "+model_type)
            vl = SocialLSTM(hidden_dim=h_dim, mediate_dim=32, output_dim=2)            
    else:
        print("reading model "+model_type)
        vl = model
    vl.to(device)

    #define loss & optimizer
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(vl.parameters(), weight_decay=0.0005)    

    plot_data = {}
    for file in files:
        plot_data[file] = [[] for _ in range(500)]

    for epoch in range(EPOCH):
        print(f"epoch {epoch+1}/{EPOCH}  ")

        for file in files:
            print(f"training on {file}")                
            #try to train this
            dataset = FramesDataset(file)

            for batch_idx, data in enumerate(dataset):
                h = torch.zeros(data['seq'].shape[1], h_dim, device=device)
                c = torch.zeros(data['seq'].shape[1], h_dim, device=device)

                print(f"batch {batch_idx+1}/{len(dataset)} ", end='\r')
                with torch.autograd.set_detect_anomaly(True):
                    Y = data['seq'][:T_pred,:,2:].clone()
                    input_seq = data['seq'][:T_pred,:,2:].clone()
                    input_seq4 = data['seq'][:T_pred,:,:].clone()
                    part_masks = data['mask']

                    #forward prop
                    if model_type == 'v':
                        output = vl(input_seq, part_masks, h, c, Y, T_obs, T_pred)
                    else:
                        # catch the coords
                        coords = []
                        for t in range(input_seq.shape[0]):
                            coord = []
                            for traj_idx in range(input_seq.shape[1]):
                                coord.append(dataset.getCoordinates(input_seq4[t,traj_idx,0].item(),
                                                                    input_seq4[t,traj_idx,1].item()))
                            coords.append(coord)
                        coords = torch.tensor(coords, device=device)
                        # coords = data['coords'][:T_pred,:,2:]
                        output = vl(input_seq, coords, part_masks, h, c, Y, T_obs, T_pred)
                    # output = vl(input_seq, pr_masks, h, c, Y, T_obs, T_pred)

                    #compute loss
                    Y_pred = output[T_obs+1:T_pred]
                    Y_g = Y[T_obs+1:T_pred]

                    cost = criterion(Y_pred, Y_g)
                    # print(f"c {criterion(Y_pred, Y_g)}, s {strideReg(Y_pred, Y_g)}")

                    if epoch % 5 == 4:
                        print(epoch, batch_idx, cost.item())

                    #save data for plotting
                    # plot_data[file][batch_idx].append(cost.item())
                    plot_data[file][batch_idx].append(cost)

                    #backward prop
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

    toc = time.time()
    print(f"training consumed {toc-tic}")

    #plot cost
    print("removing old pics")
    filelist = [f for f in listdir('eth_plots') if f.endswith(".png") ]
    for f in filelist:
        remove(join('eth_plots', f))

    for j, (k, v) in enumerate(plot_data.items()):
        printPics(v,j)

    #save the model
    torch.save(vl, name)
    print(f"saved model in {name}\n")    

    return vl


# %%
def validate(model, T_obs, T_pred, file, model_type='v'):
    #try to validate this
    h_dim = 128
    dataset = FramesDataset(file, special=True)    
    
    plotting_batches = np.arange(20)
    plotting_data = []
    avgDispErrMeans = []
    finalDispErrMeans = []    
    #validate the model based on the dataset
    print(f"validating on {file} {model_type}")
    for batch_idx, data in enumerate(dataset):
        print(f"batch  {batch_idx}", end='\r')
        traj_num = data['seq'].shape[1]
        h = torch.zeros(data['seq'].shape[1], h_dim, device=device)
        c = torch.zeros(data['seq'].shape[1], h_dim, device=device)

        if data['seq'].shape[2] > 2:
            Y = data['seq'][:T_pred,:,2:].clone()
            input_seq = data['seq'][:T_pred,:,2:].clone()
            input_seq4 = data['seq'][:T_pred,:,:].clone()
        else:
            Y = data['seq'][:T_pred,:].clone()
            input_seq = data['seq'][:T_pred,:].clone()            
        part_masks = data['mask']      
        coords = data['coords']      
        with torch.no_grad():         
            print(f"batch {batch_idx+1}/{len(dataset)}  ", end='\r')
            
            #forward prop
            if model_type == 'v':
                output = model(input_seq, part_masks, h, c, Y, T_obs, T_pred)
            else:
                if not dataset.special:
                    #catch the coords
                    coords = []
                    for t in range(input_seq.shape[0]):
                        coord = []
                        for traj_idx in range(input_seq.shape[1]):
                            coord.append(dataset.getCoordinates(input_seq4[t,traj_idx,0].item(),
                                                             input_seq4[t,traj_idx,1].item()))
                        coords.append(coord)
                    coords = torch.tensor(coords, device=device)
                else:
                    coords = data['coords'][:T_pred]
                output = model(input_seq, coords, part_masks, h, c, Y, T_obs, T_pred)
                    

            #compute cost
            Y_pred = output[T_obs+1:T_pred]
            Y_g = Y[T_obs+1:T_pred]

            #save plotting data for visualization
            if batch_idx in plotting_batches:
                # plotting_data.append((Y_pred, part_masks, traj_num, batch_idx, dataset.getCoordinates(data['seq']), T_obs, True))
                plotting_data.append((Y_pred, data['seq'][:T_pred].clone(), dataset, T_obs, False, batch_idx))                    

            if batch_idx in range(len(dataset)):
                err = ADE(Y_pred, Y_g)
                avgDispErrMeans.append(err)
                print(f"ade {err}")

            if batch_idx in range(len(dataset)):
                err = FDE(Y_pred, Y_g)
                finalDispErrMeans.append(err)            
                print(f"fde {err}")        

    for i, d in enumerate(plotting_data):
        print(f"plotting {i}th pic  ", end='\r')
        plotting_batch(*d)
        
    print("total avg disp mean ", np.sum(np.array(avgDispErrMeans))/len([v for v in avgDispErrMeans if v != 0]))
    print("total final disp mean ", np.sum(np.array(finalDispErrMeans))/len([v for v in finalDispErrMeans if v != 0]))    



def validateNew(model, T_obs, T_pred, file, start, end, model_type='v'):
    #try to validate this
    h_dim = 128
    dataset = FramesDataset(file, special=True)    
    
    plotting_batches = np.arange(20)
    plotting_data = []
    avgDispErrMeans = []
    finalDispErrMeans = []    
    #validate the model based on the dataset
    print(f"validating on {file} {model_type}")
    for batch_idx, data in enumerate(dataset):
        print(f"b  {batch_idx}")

        result_coords = torch.zeros(40, 86318, 2)
        for traj in range(start,end):
            if traj % 100 == 99:
                print(f"dealing with {traj}")

            Y1 = data['seq'][:T_pred,traj,:].clone()
            Y = data['seq'][:T_pred,traj,:].clone().reshape(Y1.shape[0], 1, Y1.shape[1])
            input_seq = data['seq'][:T_pred,traj,:].clone().reshape(Y1.shape[0], 1, Y1.shape[1]) 
            part_masks = torch.ones(input_seq.shape[0],1,1)
            # pdb.set_trace()
            coords1 = data['coords'][:T_pred,traj,:].clone() 
            coords = data['coords'][:T_pred,traj,:].clone().reshape(coords1.shape[0], 1, coords1.shape[1])    

            traj_num = input_seq.shape[1]
            h = torch.zeros(1, h_dim, device=device)
            c = torch.zeros(1, h_dim, device=device)

            with torch.no_grad():         
                print(f"batch {batch_idx+1}/{len(dataset)}  ", end='\r')

                #forward prop
                if model_type == 'v':
                    output = model(input_seq, part_masks, h, c, Y, T_obs, T_pred)
                else:
                    coords = coords
                    output = model(input_seq, coords, part_masks, h, c, Y, T_obs, T_pred)

                #save result
                result_coords[:,traj,:] = calcCoordinatesNew(input_seq[0], output)

                #compute cost
                Y_pred = output[T_obs+1:T_pred]
                Y_g = Y[T_obs+1:T_pred]

                #save plotting data for visualization
                if batch_idx in plotting_batches:
                    # plotting_data.append((Y_pred, part_masks, traj_num, batch_idx, dataset.getCoordinates(data['seq']), T_obs, True))
                    plotting_data.append((Y_pred, data['seq'][:T_pred].clone(), dataset, T_obs, False, batch_idx))                    

                if batch_idx in range(len(dataset)):
                    err = ADE(Y_pred, Y_g)
                    avgDispErrMeans.append(err)

                if batch_idx in range(len(dataset)):
                    err = FDE(Y_pred, Y_g)
                    finalDispErrMeans.append(err)            
        
    ade = np.sum(np.array(avgDispErrMeans))/len([v for v in avgDispErrMeans if v != 0])
    fde = np.sum(np.array(finalDispErrMeans))/len([v for v in finalDispErrMeans if v != 0])
    print(f"writing {start}-{end} {ade} {fde} to results.txt")
    with open("results.txt",'a') as f:
        f.write(str(start)+"-"+str(end)+": "+str(ade)+" "+str(fde)+"\n")
    print("total avg disp mean ", ade)
    print("total final disp mean ", fde)    

    # print(f"saving to {start}-{end}")
    # torch.save(result_coords,"re_trajs/"+str(start)+"-"+str(end))

    return avgDispErrMeans, finalDispErrMeans


# %%
def parse_args():
    ''' 
    python3 main.py "s" --dataset "eth" --epoch 3 '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--special_file", default='', type=str)
    parser.add_argument("--special_model", default='', type=str)    
    parser.add_argument("--special_start", default=None, type=int)    
    parser.add_argument("--dataset", default="eth", type=str)
    parser.add_argument("--T_obs", default=8, type=int)
    parser.add_argument("--T_pred", default=20, type=int)    
    parser.add_argument("--epoch", default=25, type=int)
    parser.add_argument("--model_name", default="a_just_trained_model_for_")
    parser.add_argument("model_type", type=str)
    parser.add_argument("--pure_val_name", default='', type=str)
    return parser.parse_args()


def main():
    #set device
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device {device}\n")

    #read args
    args = parse_args()

    if args.special_file != "" and args.special_model != "":
        start = args.special_start*550
        end = start+550
        m = torch.load(args.special_model)
        print(f"doing {start}-{end} with {args.special_model}")
        ade, fde = validateNew(m, args.T_obs, args.T_pred, args.special_file, start, end, model_type=args.model_type)
        return

    if args.pure_val_name == '':
        #train loop
        files_dir = join("datasets", args.dataset, "train")
        name = args.model_name+args.dataset+'.pt'
        print(f"pulling from dir {files_dir}")
        files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
        #training
        m = train(args.T_obs, args.T_pred, files, model_type=args.model_type, name=name, EPOCH=args.epoch)
    else:
        name = args.pure_val_name

    #validate loop
    torch.cuda.empty_cache()    
    m = torch.load(name)
    print(f"loading from {name}")
    #preparing validating set
    files_dir = join("datasets", args.dataset, "test")
    print(f"pulling from dir {files_dir}")        
    files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
    #validating
    for file in files:
        validate(m, args.T_obs, args.T_pred, file, model_type=args.model_type)     

# %%
if __name__ == "__main__":
    main()
    
