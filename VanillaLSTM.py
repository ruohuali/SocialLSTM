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
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from copy import deepcopy 
import pickle
from SocialLSTM import *


# %%
class FramesDataset(Dataset):
    def text2Tensor(self, file_data):
        #process the file data such that it's a list of lists of offset tuple in each time step
        file_data_t = []
        data_temp = []
        try:
            frame_num = file_data[0][0]
        except IndexError:
            print("???:")
            print(file_data)
        traj_list = []
        frame_list = []
        for line in file_data:
            if frame_num != line[0]:
                frame_num = line[0]
                data_temp.sort(key=lambda data : data[1])
                file_data_t.append(data_temp)
                data_temp = [line]
            else:    
                data_temp.append(line)
            #keep a traj list for all trajs
            if line[1] not in traj_list:
                traj_list.append(line[1])
            if line[0] not in frame_list:
                frame_list.append(line[0])
        traj_list.sort()
        frame_list.sort()  
        
        #get participants in each frame
        #@note here the elements are ped's index in the traj list
        participants = [[] for i in range(len(file_data_t))]
        for frame_idx, line in enumerate(file_data_t):
            for traj_idx, traj in enumerate(traj_list):
                in_flag = False
                for data in line:
                    if data[1] == traj:
                        in_flag = True
                        participants[frame_idx].append(traj_list.index(data[1]))
                if not in_flag:
                    file_data_t[frame_idx].append([frame_list[frame_idx], traj, 0., 0.])
            file_data_t[frame_idx].sort(key=lambda data : data[1])

        file_data_tensors = torch.tensor(file_data_t, device=device)
        
        participant_masks = []
        for frame_idx, line in enumerate(participants):
            participant_masks.append([[torch.tensor(1.) if i in participants[frame_idx] else torch.tensor(0.) for i in range(len(traj_list))]])
        participant_masks = torch.tensor(participant_masks, device=device)
        
        return traj_list, participant_masks, file_data_tensors              
    
    
    '''
    @func preprocess
    @param path: relative path for the raw data
    @note raw data~ col1: frame index, col2: traj index, (col3, col4): (y, x)
    @return traj_list: indices for each trajactory in raw data
            participants_masks~tensor(frame num x traj num): indicate the presence of each ped at each frame
            file_data_tensors~tensor(frame num x traj num x 4): the position of each traj at each frame
                                                                if not present default to (0,0)
    '''
    def preprocessBatch(self, file_data):
        file_data = sorted(file_data, key=lambda data : data[1])
        file_data_sort = sorted(file_data, key=lambda data : data[0])
        
        #turn the file into time-major multidimensional tensor
        traj_list, participant_masks, coord_tensors = self.text2Tensor(file_data_sort)
        
        #process the file data such that it contains the offsets not global coords
        file_data_off = []
        for i, line in enumerate(file_data):
            if i > 0:
                if file_data[i][1] == file_data[i-1][1]:
                    file_data_off.append([file_data[i][0], file_data[i][1], file_data[i][2]-file_data[i-1][2], file_data[i][3]-file_data[i-1][3]])
        file_data_off.sort(key=lambda data : data[0])        
        
        traj_list, participant_masks, off_tensors = self.text2Tensor(file_data_off)
        
        return traj_list, participant_masks, off_tensors, coord_tensors


    def IdLongTrajs(self, length):
        file_data = deepcopy(self.source_file)
        file_data = sorted(file_data, key=lambda data : data[1])
        #now file_data is original file data with sorted by traj
        time_stamps = []
        temp_traj_idx = file_data[0][1]
        counter = 0
        (t_s, t_e) = (file_data[0][0], file_data[0][0])
        for i, line in enumerate(file_data):
            if line[1] != temp_traj_idx:
                if counter >= length:
                    t_e = line[0]
                    time_stamps.append((temp_traj_idx, t_s, file_data[i-1][0]))
                temp_traj_idx = line[1]
                counter = 0
                t_s = line[0]
            else:
                counter+=1
        return time_stamps


    def storeFile(self,path):
        #open the file as it is
        file_data = []
        with open(path, 'r') as file:
            for line in file:
                line_data = [int(float(data)) if i < 2 else float(data) for i, data in enumerate(line.rsplit())]
                line_data[2], line_data[3] = line_data[3], line_data[2]
                file_data.append(line_data)
        return file_data

    
    def loadFileTime(self, t_s, t_e):
        file_data = deepcopy(self.source_file)
        file_data = sorted(file_data, key=lambda data : data[0])
        ret_data = []
        for line in file_data:
            if t_s <= line[0] <= t_e:
                ret_data.append(line)
        return ret_data


    def preprocess(self, length, ratio):
        #find out the time step in source file
        file_data = deepcopy(self.source_file)
        file_data = sorted(file_data, key=lambda data : data[0])
        for i in range(len(file_data)-1):
            time_step = file_data[i+1][0]-file_data[i][0]
            if time_step != 0:
                break
        self.time_step = time_step

        time_stamps = self.IdLongTrajs(length*ratio)

        for (traj_idx, t_s, t_e) in time_stamps:
            traj_file_data = self.loadFileTime(t_s, t_s+(length+1)*time_step)
            traj_list, participant_masks, off_data, coord_data = self.preprocessBatch(traj_file_data)
            data_packet = { "traj_list":traj_list,
                            "mask":participant_masks,
                            "seq":off_data,
                            "coords":coord_data }
            self.data_packets.append(data_packet)

    
    def backCompatible(self, path):
        file_data = deepcopy(self.source_file)
        self.traj_list, self.participant_masks, self.off_data, self.coord_data = self.preprocessBatch(file_data)


    def dealWithSpecial(self, file):
        with open(file, 'rb') as f:
            ft = pickle.load(f)
        #file data tensor ~ (40x86318x2)
        #get offsets
        off_data = torch.zeros(ft.shape[0]-1,ft.shape[1],ft.shape[2])
        for t in range(ft.shape[0]-1):
            off_data[t] = ft[t+1] - ft[t]
        coord_data = deepcopy(ft)
        participant_masks = torch.ones(ft.shape[0], 1, ft.shape[1])
        data_packet = { "mask":participant_masks,
                        "seq":off_data,
                        "coords":coord_data }
        self.data_packets = [data_packet]


    def __init__(self, path, length=20, ratio=0.9, special=False):
        special = True if ".p" in path else False
        if not special:
            self.source_file = self.storeFile(path)
            self.data_packets = []
            self.preprocess(length, ratio)
            self.backCompatible(path)
            self.trajs_coords = self.gatherCoordinates()
        else:
            self.dealWithSpecial(path)


    def __len__(self):
        return len(self.data_packets)
    

    '''
    @note (X, Y) is a (file_data[idx], frame[idx+1]) pair if a single idx is provided
    a (frame[idx.start]2frame[idx.end], frame[idx.start+1]2frame[idx.end+1]) pair is provided
    if a index slice is provided
    the accompanying mask tensor follows the same rule 
    '''
    def __getitem__(self, idx):
        return self.data_packets[idx]

    
    def getTrajList(self):
        return None

    
    def getParticipants(self):
        return self.participant_mask


    def getCoordinates(self, time_stamp, ped_id):
        for line in self.trajs_coords[ped_id]:
            if line[0] == time_stamp and line[1] == ped_id:
                ret_coord = (line[2],line[3])
                return ret_coord
        # print(f"{time_stamp, ped_id} no coord found")
        return (0., 0.)


    def gatherCoordinates(self):
        file_data = deepcopy(self.source_file)
        file_data = sorted(file_data, key=lambda data : data[1])
        #now file_data is original file data with sorted by traj
        trajs_coords = {}
        traj_coords = []
        temp_traj_idx = file_data[0][1]
        for i, line in enumerate(file_data):
            if line[1] != temp_traj_idx or i == len(file_data)-1:
                trajs_coords[temp_traj_idx] = traj_coords
                temp_traj_idx = line[1]
                traj_coords = []
            else:
                traj_coords.append(line)
        return trajs_coords
        

# if __name__ == '__main__':
#     D = FramesDataset("x_all.p", special=True)
#     pdb.set_trace()
#     for i in range(len(D)):
#         data_packet = D[i]
#         pdb.set_trace()


# %%
class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, l1_dim=6, l2_dim=32, l3_dim=64, l4_dim=100):
        super(LinearNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, l1_dim)
        self.lin2 = nn.Linear(l1_dim, l2_dim)
        self.lin3 = nn.Linear(l2_dim, l3_dim)
        self.lin4 = nn.Linear(l3_dim, l4_dim)
        self.lin5 = nn.Linear(l4_dim, output_dim)

    def forward(self, x):
        return self.lin5(self.lin4(self.lin3(self.lin2(self.lin1(x)))))


class Phi(nn.Module):
    ''' a non-linear layer'''
    def __init__(self, dropout_prob):
        super(Phi, self).__init__()
        self.dropout_prob = dropout_prob
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        return self.Dropout(self.ReLU(x))


class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=20, mediate_dim=10, output_dim=2, dropout_prob=0):
        super(VanillaLSTM, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.InputLayer = nn.Linear(input_dim, mediate_dim)
        self.LSTMCell = nn.LSTMCell(mediate_dim, hidden_dim)
        self.OutputLayer = nn.Linear(hidden_dim, output_dim)
        self.Phi = Phi(dropout_prob=dropout_prob)
        self.LinearNet = LinearNet(input_dim=input_dim, output_dim=mediate_dim)


    def forward(self, X, part_masks, all_h_t, all_c_t, Y, T_obs, T_pred):
        outputs = torch.empty(X.shape[0], X.shape[1], self.output_dim, device=device)
        for frame_idx, x in enumerate(X):      
            if frame_idx > T_pred:
                outputs[frame_idx] = torch.zeros(X.shape[1], self.output_dim)
                continue
                
            elif frame_idx <= T_obs:
                r = self.Phi(self.InputLayer(x))
                # r = self.LinearNet(x[:,2:])
                all_h_t, all_c_t = self.LSTMCell(r, (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
                
            elif frame_idx > T_obs and frame_idx <= T_pred:
                r = self.Phi(self.InputLayer(outputs[frame_idx-1].clone()))
                # r = self.LinearNet(outputs[frame_idx-1].clone())
                all_h_t, all_c_t = self.LSTMCell(r, (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask                
                
            #dirty fix for appearance that's too short
            if frame_idx > 3 and frame_idx > T_obs:
                for traj_idx in torch.where(part_masks[frame_idx][0] != 0)[0]:
                    if part_masks[frame_idx-3][0][traj_idx] == 0:
                        outputs[frame_idx, traj_idx] = Y[frame_idx, traj_idx] 

        return outputs


# %%
def trajPruningByAppear(part_mask, ratio=0.6, in_tensor=None):
    if in_tensor != None:
        #count appearance
        for traj in range(in_tensor.shape[1]):
            traj_mask = part_mask[:,0,traj]
            count = traj_mask[traj_mask!=0].shape[0]
            if count < part_mask.shape[0]*ratio:
                in_tensor[:,traj,:] *= 0.
        return in_tensor
    else:
        new_mask = part_mask.clone()
        #count appearance
        for traj in range(part_mask.shape[2]):
            traj_mask = part_mask[:,0,traj]
            count = traj_mask[traj_mask!=0].shape[0]
            if count < part_mask.shape[0]*ratio:
                new_mask[:,0,traj] *= 0.        
        return new_mask
    
'''@note last elem in in_tensors must be the mask'''
def trajPruningByStride(part_mask, ref_tensor, in_tensors, length=0.3):
    for traj in range(ref_tensor.shape[1]):
        actual_strides = ref_tensor[:,traj,2:]
        avg_stride_len = torch.mean(torch.abs(actual_strides[actual_strides!=torch.zeros(2, device=device)]))
        #if divide by zero means the traj never appears in the batch
        if math.isnan(avg_stride_len):
            for i, in_tensor in enumerate(in_tensors):
                if i == len(in_tensors)-1:
                    in_tensors[i][:,0,traj] *= 0.
                else:
                    in_tensors[i][:,traj,:] *= 0.   
        if avg_stride_len < length:
            for i, in_tensor in enumerate(in_tensors):
                if i == len(in_tensors)-1:
                    in_tensors[i][:,0,traj] *= 0.
                else:
                    in_tensors[i][:,traj,:] *= 0.                    
    return in_tensors


def strideReg(X, Y):
    X_all = X.reshape(X.shape[0]*X.shape[1],X.shape[2])
    Y_all = Y.reshape(Y.shape[0]*Y.shape[1],Y.shape[2])    
    X_total_len = torch.sum(torch.abs(X_all))
    Y_total_len = 1.5*torch.sum(torch.abs(Y_all))
    Loss = torch.abs(Y_total_len - X_total_len)
    return Loss


# %%
def train(T_obs, T_pred, files, model_type='v', model=None, name="model.pt"):
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

    EPOCH = 5
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

                    #dirty truncate
                    # run_ratio = (T_obs+2)/T_pred
                    # input_seq = trajPruningByAppear(part_masks, ratio=run_ratio, in_tensor=input_seq) 
                    # Y = trajPruningByAppear(part_masks, ratio=run_ratio, in_tensor=Y)     
                    # pr_masks = trajPruningByAppear(part_masks, ratio=run_ratio)         
                    # (input_seq, Y, pr_masks) = trajPruningByStride(pr_masks, input_seq, (input_seq, Y, pr_masks))   
                    
                    #forward prop
                    if model_type == 'v':
                        output = vl(input_seq, part_masks, h, c, Y, T_obs, T_pred)
                    else:
                        #catch the coords
                        coords = []
                        for t in range(input_seq.shape[0]):
                            coord = []
                            for traj_idx in range(input_seq.shape[1]):
                                coord.append(dataset.getCoordinates(input_seq4[t,traj_idx,0].item(),
                                                                    input_seq4[t,traj_idx,1].item()))
                            coords.append(coord)
                        coords = torch.tensor(coords, device=device)
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
    def ppp(v,j):
        plt.figure()
        plt.title(str(vl.hidden_dim))
        for i, data in enumerate(v):
            if len(data) != 0:
                plt.plot(np.arange(len(v[0])), data)        
        plt.savefig("eth_plots/"+"train"+str(j))
        print("--->","eth_plots/"+"train"+str(j))
    j = 0
    for k, v in plot_data.items():
        #print(f"k {k} \n v {v[0]} {len(v)}")
        ppp(v,j)
        j+=1
    
    # plot_data = np.array(plot_data)
    # avg_plot_data = np.mean(plot_data, axis=0)
    # plt.figure()
    # plt.plot(avg_plot_data)
    # plt.savefig("eth_plots/"+"train"+str(i)+"avg")
    

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
            #dirty truncate
            # run_ratio = (T_obs+2)/T_pred
            # input_seq = trajPruningByAppear(part_masks, ratio=run_ratio, in_tensor=input_seq) 
            # Y = trajPruningByAppear(part_masks, ratio=run_ratio, in_tensor=Y)     
            # pr_masks = trajPruningByAppear(part_masks, ratio=run_ratio)
            # (input_seq, Y, pr_masks) = trajPruningByStride(pr_masks, input_seq, (input_seq, Y, pr_masks))
            
            #forward prop
            if model_type == 'v':
                output = model(input_seq, part_masks, h, c, Y, T_obs, T_pred)
            else:
                #catch the coords
                coords = []
                for t in range(input_seq.shape[0]):
                    coord = []
                    for traj_idx in range(input_seq.shape[1]):
                        coord.append(dataset.getCoordinates(input_seq4[t,traj_idx,0].item(),
                                                            input_seq4[t,traj_idx,1].item()))
                    coords.append(coord)
                coords = torch.tensor(coords, device=device)
                output = model(input_seq, coords, part_masks, h, c, Y, T_obs, T_pred)


            #compute cost
            Y_pred = output[T_obs+1:T_pred]
            Y_g = Y[T_obs+1:T_pred]
            #......
            #get and process result                
            # Y_pred_param = Y_pred.clone()
            # coords_param = dataset.getCoordinates(data['seq']).clone()

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
        
    # for i, d in enumerate(plotting_data):
    #     print(f"plotting {i}th pic")
    #     plotting_batch(*d)
        
    print("total avg disp mean ", np.sum(np.array(avgDispErrMeans))/len([v for v in avgDispErrMeans if v != 0]))
    print("total final disp mean ", np.sum(np.array(finalDispErrMeans))/len([v for v in finalDispErrMeans if v != 0]))    


# %%
def ADE(X, Y):
    result = 0.
    for traj_idx in range(X.shape[1]):
        dist = 0.
        pos_X, pos_Y = torch.tensor([0.,0.], device=device), torch.tensor([0.,0.], device=device)
        for t in range(X.shape[0]):
            pos_X += X[t,traj_idx]
            pos_Y += Y[t,traj_idx]
            dist += torch.dist(pos_Y, pos_X)
        dist /= X.shape[0]
        result += dist
    result /= X.shape[1]
    print(f"avg disp error {result}")
    return result    


def FDE(X, Y):
    result = 0.
    for traj_idx in range(X.shape[1]):
        dist = 0.
        pos_X, pos_Y = torch.tensor([0.,0.], device=device), torch.tensor([0.,0.], device=device)
        for t in range(X.shape[0]):
            pos_X += X[t,traj_idx]
            pos_Y += Y[t,traj_idx]
        dist += torch.dist(pos_Y, pos_X)
        result += dist
    result /= X.shape[1]
    print(f"final disp error {result}")
    return result        


# %%
def plotting_batch(batch_trajs_pred_gpu, input_seq, dataset, T_obs, is_total, batch_idx):          
    #reform the trajs tensor to a list of each traj's pos at each frame
    batch_trajs_pred = batch_trajs_pred_gpu.cpu().data.numpy()
    trajs_pred_list = [[] for _ in range(batch_trajs_pred.shape[1])]
    part = []
    for traj_idx in range(batch_trajs_pred.shape[1]):
        if np.sum(batch_trajs_pred[:,traj_idx,:]) == 0.:
            continue
        trajs_pred_list[traj_idx] = batch_trajs_pred[:,traj_idx,:]    
        part.append(traj_idx)

    trajs_coord_list = [[] for _ in range(batch_trajs_pred.shape[1])]
    for traj_idx in part:
        for t in range(input_seq.shape[0]):
            if (input_seq[t,traj_idx,2:] == torch.tensor([0.,0.],device=device)).all():
                trajs_coord_list[traj_idx].append((0.,0.))   
                continue  
            trajs_coord_list[traj_idx].append(dataset.getCoordinates(input_seq[t,traj_idx,0].item(),input_seq[t,traj_idx,1].item()))
        trajs_coord_list[traj_idx].append(dataset.getCoordinates(input_seq[t,traj_idx,0].item()+dataset.time_step,input_seq[t,traj_idx,1].item()))
    
    #calc the coords of each step
    trajs_pred_coords = np.zeros((len(trajs_pred_list), input_seq.shape[0]-T_obs, 2))
    for traj_idx, traj in enumerate(trajs_pred_list):
        if traj_idx not in part:
            continue
        split_point = np.array(trajs_coord_list[traj_idx][T_obs+1])

        trajs_pred_coords[traj_idx,0] += np.array(split_point)
        next_point = split_point
        for i, off in enumerate(traj):
            next_point += np.array(off)
            trajs_pred_coords[traj_idx,i+1] = next_point
    # trajs_pred_coords = np.array(trajs_pred_coords)
        
    #plot
    plt.figure(figsize=(30,30))
    plt.xlim([-15,15])
    plt.ylim([-15,15])
    plot_idx = 0
    for traj_idx in part:
        try:
            pred_x = trajs_pred_coords[traj_idx][:,0]
        except IndexError:
            print("not enough appearance")
            continue
        pred_y = trajs_pred_coords[traj_idx][:,1]            
        plt.plot(pred_x, pred_y, label="pred"+str(traj_idx), marker='.')
        for i, (x, y) in enumerate(zip(pred_x, pred_y)):
            if i < len(pred_x)-1:
                try:
                    plt.arrow(x, y, (pred_x[i+1] - x)/2, (pred_y[i+1] - y)/2, width=0.0001, head_width=0.001, head_length=0.001)    
                except IndexError:
                    print("plot error")

        trajs_coord = np.array(trajs_coord_list[traj_idx])
        total_x = trajs_coord[:,0]    
        total_x = total_x[np.where(total_x != 0.)]
        total_y = trajs_coord[:,1]
        total_y = total_y[np.where(total_y != 0.)]       
        try:
            plt.plot(total_x, total_y, linestyle="dashed", label="total"+str(traj_idx), marker='1')
        except ValueError:
            print("plot error")
            
        for i, (x, y) in enumerate(zip(total_x, total_y)):
            if i < len(total_x)-1:
                try:
                    plt.arrow(x, y, (total_x[i+1] - x)/2, (total_y[i+1] - y)/2, width=0.0001, head_width=0.001, head_length=0.001)
                except IndexError:
                    print("plot error")
        plot_idx += 1
 
    plt.legend(loc="upper right")
    plt.title(f"batch {batch_idx}")
    plt.savefig("eth_plots/"+str(batch_idx)+str(is_total)+"6-6-20-all")    



# %%
if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device {device}\n")

#     # train_datasets = ["datasets/eth/train",
#     #                   "datasets/hotel/train",               
#     #                   "datasets/univ/train",
#     #                   "datasets/zara1/train",
#     #                   "datasets/zara2/train"
#     #                  ]
#     # val_datasets = ["datasets/eth/test",
#     #                 "datasets/hotel/test",               
#     #                 "datasets/univ/test",
#     #                 "datasets/zara1/test",
#     #                 "datasets/zara2/test"
#     #                 ]
#     # names = ["eth_vl.pt",
#     #          "hotel_vl.pt",
#     #          "univ_vl.pt",
#     #          "zara1_vl.pt",
#     #          "zara2_vl.pt"
#     #         ]
    
#     # for train_dataset, val_dataset, name in zip(train_datasets, val_datasets, names):
#     #     #preparing training set
#     #     files_dir = train_dataset
#     #     print(f"pulling from dir {files_dir}")
#     #     files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
#     #     vl = None
#     #     #training
#     #     for file in files:
#     #         vl = train(8, 20, file, model=vl, name=name)

# #         vl1 = torch.load(name)
# #         print(f"loading from {name}")
# #     #     validate(vl1, 8, 20, "try_dataset.txt")       
# #         #preparing validating set
# #         files_dir = val_dataset
# #         print(f"pulling from dir {files_dir}")        
# #         files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
# #         #validating
# #         for file in files:
# #             validate(vl1, 8, 20, file)   
            
# #         print("====================================")

    # files_dir = "datasets/eth/train"
    # name = "eth_vl.pt"
    # print(f"pulling from dir {files_dir}")
    # files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
    # #training
    # vl = train(8, 20, files, name=name)

    # torch.cuda.empty_cache()    
    # vl1 = torch.load("eth_vl.pt")
    # print(f"loading from eth_vl.pt")
    # #preparing validating set
    # files_dir = "datasets/eth/test"
    # print(f"pulling from dir {files_dir}")        
    # files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
    # #validating
    # for file in files:
    #     validate(vl1, 8, 20, file) 

    ###############################################
    files_dir = "datasets/eth/train"
    name = "eth_sl.pt"
    print(f"pulling from dir {files_dir}")
    files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
    #training
    vl = train(8, 20, files, name=name, model_type='s')

    torch.cuda.empty_cache()    
    vl1 = torch.load("eth_vl.pt")
    print(f"loading from eth_vl.pt")
    #preparing validating set
    files_dir = "datasets/eth/test"
    print(f"pulling from dir {files_dir}")        
    files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
    #validating
    for file in files:
        validate(vl1, 8, 20, file, model_type='s') 
    #################################################

    # temp = train(8, 20, ["datasets/hotel/test/biwi_hotel.txt"], model_type='s')
#     # validate(temp, 8, 20, "datasets/eth/test/biwi_eth.txt")
#     # temp = torch.load("model.pt")
#     # validate(temp, 8, 20, "datasets/hotel/test/biwi_hotel.txt")
#     # validate(temp, 8, 20, "datasets/eth/test/biwi_eth.txt")

    # temp = train(8, 20, ["try_dataset.txt"], model_type='s')
    # validate(temp, 8, 20, "try_dataset.txt", model_type='s')

    # validate(vl1, 20, 40, "x_all.p")
