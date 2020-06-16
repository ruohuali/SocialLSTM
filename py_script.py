import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from scipy.stats import multivariate_normal

from torch.utils.data import Dataset, DataLoader
import pandas

#=====================================================================================================

def preprocess(path):
    file_data = []
    with open(path, 'r') as file:
        for line in file:
            line_data = [int(float(data)) if i < 2 else float(data) for i, data in enumerate(line.rsplit())]
            file_data.append(line_data)
    file_data.sort(key=lambda data : data[0])

    file_data_t = []
    data_temp = []
    frame_num = file_data[0][0]
    for line in file_data:
        if frame_num != line[0]:
            frame_num = line[0]
            file_data_t.append(data_temp)
            data_temp = [line]
        else:    
            data_temp.append(line)

    file_data_batch = []
    peds = [data[1] for data in file_data_t[0]]
    batch = []
    for line in file_data_t:
        new_peds = [data[1] for data in line]
        if new_peds != peds:
            peds = new_peds
            file_data_batch.append(batch)
            batch = [line]
        else:
            batch.append(line)

    file_data_tensors = []
    for i, line in enumerate(file_data_batch):
        data_tensors = []
        for j, t_inst in enumerate(file_data_batch[i]):
            data_tensor = [[t_inst[k][3],t_inst[k][2]] for k in range(len(t_inst))]
            data_tensors.append(data_tensor)
        data_tensors = torch.Tensor(data_tensors)
        file_data_tensors.append(data_tensors)
    
    return file_data_tensors


class FramesDataset(Dataset):
    def __init__(self, path):
        self.file_data = preprocess(path)

    def __len__(self):
        return len(self.file_data)

    def __getitem__(self, idx):
        data = self.file_data[idx].clone()
        return data[:data.shape[0]-1], data[1:]


'''==========================================================================================================='''



def Gaussian2D(params, y):
    (mu_x,mu_y), (sig_x,sig_y), rho_xy = (params[0],params[1]), (params[2],params[3]), params[4]
    # (mu_x,mu_y), (sig_x,sig_y), rho_xy = (params[0],params[1]), (2, 2), 0
    covariance = rho_xy*sig_x*sig_y
    # rv = MultivariateNormal(torch.Tensor([mu_x, mu_y]), torch.Tensor([[sig_x, covariance], [covariance, sig_y]]))
    
    
    logP = 
    logP.requires_grad = True
    return logP


def Gaussian2DNll(all_params, targets):
    traj_num = targets.shape[1]
    T = targets.shape[0]

    L = torch.zeros(traj_num)
    for traj in range(traj_num):
        for t in range(T):
            L[traj] += Gaussian2D(all_params[t][traj], targets[t][traj])
    L *= -1

    cost = torch.sum(L)

    return cost

'''==========================================================================================================='''


class Phi(nn.Module):
    ''' a non-linear layer'''
    def __init__(self, dropout_prob):
        super(Phi, self).__init__()
        self.dropout_prob = dropout_prob
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        return self.Dropout(self.ReLU(x))


class SocialLstm(nn.Module):
    '''r is input embedding dim e is social pooling embedding dim (r_dim+e_dim) is embedding dim'''
    def __init__(self, input_dim=2, r_dim=100, e_dim=100, hidden_dim=128,
     output_dim=5, dropout_prob=0.1, grid_cell_size=5, N_size=5):
        super(SocialLstm, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim        
        self.hidden_dim = hidden_dim
        self.grid_cell_size = grid_cell_size
        self.N_size = N_size
        self.InputEmbedding = nn.Linear(input_dim, r_dim)
        self.SocialEmbedding = nn.Linear((self.N_size+1)**2*self.hidden_dim, e_dim)
        self.Phi = Phi(dropout_prob)
        self.LstmCell = nn.LSTMCell(r_dim+e_dim, hidden_dim)
        # self.LstmCell = nn.LSTMCell(r_dim, hidden_dim)
        self.OutputLayer = nn.Linear(hidden_dim, output_dim)
        self.CorrNormLayer = nn.Sigmoid()


    def socialPooling(self, h_tm1, coords):
        H = torch.ones(coords.shape[0], self.N_size+1, self.N_size+1, self.hidden_dim)
        for i in range(coords.shape[0]):
            for j in range(coords.shape[0]):
                if i == j:
                    continue
                #calc relative grid coord
                grid_coord = ( ((coords[j][0]-coords[i][0]).numpy()).item() / self.grid_cell_size,
                               ((coords[j][1]-coords[i][1]).numpy()).item() / self.grid_cell_size )
                if np.abs(grid_coord[0]) <= self.N_size/2 and np.abs(grid_coord[1]) <= self.N_size/2:
                    #convert to positive for indexing
                    grid_coord = ( int(grid_coord[0]+self.N_size/2), int(grid_coord[1]+self.N_size/2) )
                    H[i][grid_coord[0]][grid_coord[1]] += h_tm1[j]
        
        H = H.reshape(coords.shape[0], (self.N_size+1)**2*self.hidden_dim)
        return H

    
    def forward(self, coords_seq, h_t, c_t):
        outputs = torch.empty(coords_seq.shape[0], coords_seq.shape[1], self.output_dim)
        #for each time-step
        for t, coords in enumerate(coords_seq):
            r = self.Phi(self.InputEmbedding(coords))
            H = self.socialPooling(h_t, coords)
            e = self.Phi(self.SocialEmbedding(H))
            concat_embed = torch.cat((r,e), 1)
            h_t, c_t = self.LstmCell(concat_embed, (h_t, c_t))
        
            y = self.OutputLayer(h_t)
            if self.output_dim == 5:
                corr = self.CorrNormLayer(y[:,4])
                outputs[t,:,:2] = y[:,:2]
                outputs[t,:,2:4] = torch.abs(y[:,2:4])
                outputs[t,:,4] = corr
            elif self.output_dim == 2:
                outputs[t] = y

        return outputs


'''==========================================================================================================='''


def train():
    #try to train this
    h_dim = 128

    dataset = FramesDataset("biwi_hotel_4.txt")
    input_seq, Y = dataset[0]
    h = torch.randn(input_seq.shape[1],h_dim)
    c = torch.randn(input_seq.shape[1],h_dim)

    sl = SocialLstm(output_dim=5, hidden_dim=h_dim, grid_cell_size=0.2, N_size=5)

    #define loss & optimizer
    # criterion = nn.MSELoss(reduction="sum")
    criterion = Gaussian2DNll
    optimizer = torch.optim.Adagrad(sl.parameters(), weight_decay=0.005)

    for epoch in range(1000):
        #forward pass
        output = sl(input_seq, h, c)

        #compute loss
        cost = criterion(output, Y)

        if epoch % 10 == 9:
            print(epoch, cost.item())

        #backward pass
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

train()