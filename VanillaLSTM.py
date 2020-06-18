# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import pdb 


# %%
def preprocess(path):
    file_data = []
    with open(path, 'r') as file:
        for line in file:
            line_data = [int(float(data)) if i < 2 else float(data) for i, data in enumerate(line.rsplit())]
            line_data[2], line_data[3] = line_data[3], line_data[2]
            file_data.append(line_data)
    file_data.sort(key=lambda data : data[0])

    file_data_t = []
    data_temp = []
    frame_num = file_data[0][0]
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

    file_data_tensors = torch.tensor(file_data_t)

    participant_masks = []
    for frame_idx, line in enumerate(participants):
        participant_masks.append([[torch.tensor(1.) if i in participants[frame_idx] else torch.tensor(0.) for i in range(len(traj_list)) ]])
    participant_masks = torch.tensor(participant_masks)

    return traj_list, participant_masks, file_data_tensors


class FramesDataset(Dataset):
    def __init__(self, path):
        self.traj_list, self.participant_masks, self.file_data = preprocess(path)

    def __len__(self):
        return len(self.file_data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < len(self.file_data)-1:
                Y_idx = idx+1
            else:
                Y_idx = len(self.file_data)-1

        else:
            if idx.start != None:
                start = idx.start+1
            else:
                start = 0+1
            if idx.stop != None:
                stop = idx.stop+1
            else:
                stop = len(self.file_data)-1
            Y_idx = slice(start, stop)

        participant_mask = self.participant_masks[idx]
        X = self.file_data[idx]
        Y = self.file_data[Y_idx]

        return participant_mask, (X, Y)

    def getTrajList(self):
        return self.traj_list

    def getParticipants(self):
        return self.participant_mask


# %%
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
    def __init__(self, input_dim=2, hidden_dim=20, mediate_dim=10, output_dim=2, traj_num=3, dropout_prob=0.2):
        super(VanillaLSTM, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.traj_num = traj_num
        self.InputLayer = nn.Linear(input_dim, mediate_dim)
        self.LSTMCell = nn.LSTMCell(mediate_dim, hidden_dim)
        self.OutputLayer = nn.Linear(hidden_dim, output_dim)
        self.Phi = Phi(dropout_prob=dropout_prob)


    def forward(self, X, part_masks, all_h_t, all_c_t, T_obs, T_pred):
        outputs = torch.empty(X.shape[0], X.shape[1], self.output_dim)
        for frame_idx, x in enumerate(X):      
            if frame_idx <= T_obs or frame_idx > T_pred:
                outputs[frame_idx] = torch.zeros(X.shape[1], self.output_dim)
                continue

            r = self.Phi(self.InputLayer(x[:,2:]))
            all_h_t, all_c_t = self.LSTMCell(r, (all_h_t, all_c_t))
            part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
            outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask

        return outputs


# %%
def train(T_obs, T_pred):
    h_dim = 100
    batch_size = 24

    #try to train this
    dataset = FramesDataset("biwi_hotel_4.txt")
    #a dataloader for now not sure how to use
    dataloader = DataLoader(dataset, batch_size=batch_size)

    traj_num = len(dataset.getTrajList())
    h = torch.zeros(traj_num, h_dim)
    c = torch.zeros(traj_num, h_dim)

    vl = VanillaLSTM(hidden_dim=h_dim, mediate_dim=10, output_dim=2, traj_num=traj_num)
    for param in vl.parameters():
        param.requires_grad = True

    #define loss & optimizer
    criterion = nn.MSELoss(reduction="sum")
    # criterion = Gaussian2DNll
    
    optimizer = torch.optim.Adagrad(vl.parameters(), weight_decay=0.0005)
    print("training")
    for epoch in range(500):
        for batch_idx, (part_masks, (input_seq, Y)) in enumerate(dataloader):
            if batch_idx < len(dataset) // batch_size:
                Y = Y[:,:,2:]
                with torch.autograd.set_detect_anomaly(True):         
                    #forward prop
                    output = vl(input_seq, part_masks, h, c, T_obs, T_pred)

                    #compute loss
                    Y_pred = output[T_obs+1:T_pred]
                    Y_g = Y[T_obs+1:T_pred]
                    cost = criterion(Y_pred, Y_g)

                    if epoch % 10 == 9:
                        print(epoch, batch_idx, cost.item())

                    #backward prop
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
    return vl


def validate(model, T_obs, T_pred):
    #try to validate this
    h_dim = 100

    dataset = FramesDataset("biwi_hotel_4.txt")
    #a dataloader for now not sure how to use
    dataloader = DataLoader(dataset, batch_size=24)

    T_start_offset = 40; batch_size = 21
    part_masks, (input_seq, Y) = dataset[T_start_offset:T_start_offset + batch_size]
    Y = Y[:,:,2:]
    traj_num = len(dataset.getTrajList())
    h = torch.zeros(traj_num, h_dim)
    c = torch.zeros(traj_num, h_dim)
    
    print("\nvalidating")
    with torch.autograd.set_detect_anomaly(True):         
        output = model(input_seq, part_masks, h, c, T_obs, T_pred)

        for batch_idx, (part_masks, (input_seq, Y)) in enumerate(dataloader):
            if batch_idx < len(dataset) // batch_size:
                Y = Y[:,:,2:]
                with torch.autograd.set_detect_anomaly(True):         
                    print("at batch", batch_idx)
                    #forward prop
                    output = vl(input_seq, part_masks, h, c, T_obs, T_pred)

                    #compute loss
                    Y_pred = output[T_obs+1:T_pred]
                    Y_g = Y[T_obs+1:T_pred]

                    for frame_idx, (y_pred, y_g) in enumerate(zip(Y_pred, Y_g)):
                        for traj_idx in range(part_masks[frame_idx].shape[1]):
                            if part_masks[frame_idx+T_obs+1][0][traj_idx] != 0:
                                dist = torch.dist(y_pred[traj_idx],y_g[traj_idx]).item()
                                print("at frame", frame_idx+T_obs+1, "ped", traj_idx, "is off by", dist)
                        print("")
                    print("================================================================")



if __name__ == "__main__":
    vl = train(12, 20)
    validate(vl, 12, 20)
