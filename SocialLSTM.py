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
import time
import matplotlib.pyplot as plt


# %%
class FramesDataset(Dataset):
    '''
    @func preprocess
    @param path: relative path for the raw data
    @note raw data~ col1: frame index, col2: traj index, (col3, col4): (y, x)
    @return traj_list: indices for each trajactory in raw data
            participants_masks~tensor(frame num x traj num): indicate the presence of each ped at each frame
            file_data_tensors~tensor(frame num x traj num x 4): the position of each traj at each frame
                                                                if not present default to (0,0)
    '''
    def preprocess(self, path):
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

        file_data_tensors = torch.tensor(file_data_t, device=device)

        participant_masks = []
        for frame_idx, line in enumerate(participants):
            participant_masks.append([[torch.tensor(1.) if i in participants[frame_idx] else torch.tensor(0.) for i in range(len(traj_list)) ]])
        participant_masks = torch.tensor(participant_masks, device=device)

        return traj_list, participant_masks, file_data_tensors
    
    def __init__(self, path):
        self.traj_list, self.participant_masks, self.file_data = self.preprocess(path)

    def __len__(self):
        return len(self.file_data)

    '''
    @note (X, Y) is a (file_data[idx], frame[idx+1]) pair if a single idx is provided
    a (frame[idx.start]2frame[idx.end], frame[idx.start+1]2frame[idx.end+1]) pair is provided
    if a index slice is provided
    the accompanying mask tensor follows the same rule 
    '''
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
        return self.participants


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


class SocialLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=20, mediate_dim=10, output_dim=2, social_dim=128, traj_num=3, dropout_prob=0.2,
                N_size=4, grid_cell_size=1):
        super(SocialLSTM, self).__init__()
        #specify params
        self.input_dim, self.mediate_dim, self.output_dim, self.hidden_dim = input_dim, mediate_dim, output_dim, hidden_dim
        self.traj_num = traj_num
        self.grid_cell_size = grid_cell_size
        self.N_size = N_size if N_size % 2 == 0 else N_size + 1      
        #specify embedding layers
        self.InputEmbedding = nn.Linear(input_dim, mediate_dim)
        self.SocialEmbedding = nn.Linear((self.N_size)**2*self.hidden_dim, social_dim)        
        self.LSTMCell = nn.LSTMCell(mediate_dim+social_dim, hidden_dim)        
        self.OutputLayer = nn.Linear(hidden_dim, output_dim)
        self.Phi = Phi(dropout_prob=dropout_prob)
        self.CorrNormLayer = nn.Sigmoid()


    def socialPooling(self, h_tm1, coords, mask):
        print("=>",end='', flush=True)
        H = torch.zeros(coords.shape[0], self.N_size, self.N_size, self.hidden_dim, device=device)
        for i in range(coords.shape[0]):
            for j in range(coords.shape[0]):
                if i == j or mask[i] == 0 or mask[j] == 0:
                    continue
                #calc relative grid coord
                grid_coord = ( int(((coords[j][0]-coords[i][0])).item() / self.grid_cell_size),
                                int(((coords[j][1]-coords[i][1])).item() / self.grid_cell_size) )
                #check if the coord is in the neighborhood
                if np.abs(grid_coord[0]) <= self.N_size/2-1 and np.abs(grid_coord[1]) <= self.N_size/2-1:
                    #convert to positive for indexing
                    grid_coord = ( int(grid_coord[0]+self.N_size/2), int(grid_coord[1]+self.N_size/2) )
                    H[i][grid_coord[0]][grid_coord[1]] += h_tm1[j]
        
        H = H.reshape(coords.shape[0], (self.N_size)**2*self.hidden_dim)
        return H


    def forward(self, X, part_masks, all_h_t, all_c_t, Y, T_obs, T_pred):
        # print(f"forward in a batch")
        outputs = torch.empty(X.shape[0], X.shape[1], self.output_dim, device=device)
        for frame_idx, x_temp in enumerate(X):      
            time1 = time.time()
            x = x_temp[:,2:]
            if frame_idx > T_pred:
                outputs[frame_idx] = torch.zeros(X.shape[1], self.output_dim)
                continue

            #calc input embedding
            r = self.Phi(self.InputEmbedding(x))
            #calc social pooling embedding
            H = self.socialPooling(all_h_t, x, part_masks[frame_idx][0])
            e = self.Phi(self.SocialEmbedding(H))
            concat_embed = torch.cat((r,e), 1)
            all_h_t, all_c_t = self.LSTMCell(concat_embed, (all_h_t, all_c_t))
            part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
            outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
            # print(f"in a for total {time3-time1} pool {time2-time1}")

            if frame_idx > 3:
                for traj_idx in torch.where(part_masks[frame_idx][0] != 0)[0]:
                    if part_masks[frame_idx-3][0][traj_idx] == 0:
                        outputs[frame_idx, traj_idx] = Y[frame_idx, traj_idx] 

        return outputs


# %%
def train(T_obs, T_pred):
    tic = time.time()

    h_dim = 128
    batch_size = T_obs + T_pred

    #try to train this
    dataset = FramesDataset("crowds_zara02.txt")
    #a dataloader for now not sure how to use
    dataloader = DataLoader(dataset, batch_size=batch_size)

    traj_num = len(dataset.getTrajList())
    h = torch.zeros(traj_num, h_dim, device=device)
    c = torch.zeros(traj_num, h_dim, device=device)

    sl = SocialLSTM(hidden_dim=h_dim, mediate_dim=100, output_dim=2, traj_num=traj_num)
    sl.to(device)

    #define loss & optimizer
    criterion = nn.MSELoss(reduction="sum")
    # criterion = Gaussian2DNll
    optimizer = torch.optim.Adagrad(sl.parameters(), weight_decay=0.0005)
    
    print("training")
    plot_data = [[] for _ in range(len(dataset) // batch_size)]
    EPOCH = 5
    for epoch in range(EPOCH):
        print(f"epoch {epoch} of {EPOCH-1}")
        for batch_idx, (part_masks, (input_seq, Y)) in enumerate(dataloader):
            if batch_idx < len(dataset) // batch_size:
                Y = Y[:,:,2:]
                with torch.autograd.set_detect_anomaly(True):         
                    #forward prop
                    output = sl(input_seq, part_masks, h, c, Y, T_obs, T_pred)

                    #compute loss
                    Y_pred = output[T_obs+1:T_pred]
                    Y_g = Y[T_obs+1:T_pred]
                    cost = criterion(Y_pred, Y_g)

                    if epoch % 10 != 9:
                        print(" ", epoch, batch_idx, cost.item())

                    #save data for plotting
                    plot_data[batch_idx].append(cost.item())

                    #backward prop
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

    toc = time.time()
    print(f"training consumed {toc-tic}")

    #plot the cost
    plt.figure()
    for data in plot_data:
        plt.plot(np.arange(len(plot_data[0])), data)
    plt.savefig("costs2.png")

    #save the model
    torch.save(sl, "sl_4_08.pt")

    return sl


def validate(model, T_obs, T_pred):
    #try to validate this
    h_dim = 128

    batch_size = T_obs + T_pred

    dataset = FramesDataset("crowds_zara02.txt")
    #a dataloader for now not sure how to use
    dataloader = DataLoader(dataset, batch_size=batch_size)

    traj_num = len(dataset.getTrajList())
    h = torch.zeros(traj_num, h_dim, device=device)
    c = torch.zeros(traj_num, h_dim, device=device)

    print("validating")
    #validate the model based on the dataset
    for batch_idx, (part_masks, (input_seq, Y)) in enumerate(dataloader):
        if batch_idx < len(dataset) // batch_size:
            Y = Y[:,:,2:]
            with torch.autograd.set_detect_anomaly(True):         
                print(f"batch {batch_idx}")
                #forward prop
                output = model(input_seq, part_masks, h, c, Y, T_obs, T_pred)

                #compute loss
                Y_pred = output[T_obs+1:T_pred]
                Y_g = Y[T_obs+1:T_pred]

                dists_list = [np.array([]) for _ in range(traj_num)]
                batch_parts = []
                for frame_idx, (y_pred, y_g) in enumerate(zip(Y_pred, Y_g)):
                    for traj_idx in range(part_masks[frame_idx].shape[1]):
                        if part_masks[frame_idx+T_obs+1][0][traj_idx] != 0:
                            if traj_idx not in batch_parts:
                                batch_parts.append(traj_idx)
                            dist = torch.dist(y_pred[traj_idx],y_g[traj_idx]).item()
                            print(f"at frame {frame_idx+T_obs+1} ped {traj_idx} is off by {dist}\n")
                            dists_list[traj_idx] = np.append(dists_list[traj_idx], dist)
                
                total_ADE = np.array([])
                total_FDE = np.array([])
                for batch_part in batch_parts:
                    ADE = np.average(dists_list[batch_part])
                    print(f"ADE of traj {batch_part} {ADE}")
                    total_ADE = np.append(total_ADE, ADE)
                    print(f"FDE of traj {batch_part} {dists_list[batch_part][-1]}")
                    total_FDE = np.append(total_FDE, dists_list[batch_part][-1])
                    print("------------------------------------------------------")

                print(f"total ADE {np.average(total_ADE)}")
                print(f"total FDE {np.average(total_FDE)}")
                print("================================================================")

                if batch_idx == 1:
                    exp_plotting_data = (Y[:T_pred], Y[:T_obs+2], (Y_pred,Y_g), part_masks, traj_num, batch_idx)
                
                torch.save(Y[:T_pred], "sl_4_1_img/"+str(batch_idx)+" one.pt")
                torch.save(Y[:T_obs+2], "sl_4_1_img/"+str(batch_idx)+" two.pt")
                torch.save(Y_pred, "sl_4_1_img/"+str(batch_idx)+" three.pt")
                torch.save(Y_g, "sl_4_1_img/"+str(batch_idx)+" four.pt")
                torch.save(part_masks, "sl_4_1_img/"+str(batch_idx)+" five.pt")

    plotting_batch(*exp_plotting_data)

'''
@param trajs~(frame_num of a batch x traj_num x 2)
'''
def plotting_batch(total_trajs, prev_trajs, batch_trajs, part_masks, traj_num, batch_idx):
    #reform the trajs tensor to a list of each traj's pos at each frame
    batch_trajs_pred = batch_trajs[0].cpu().data.numpy()
    batch_trajs_g = batch_trajs[1].cpu().data.numpy()
    trajs_pred_list = [np.array([]) for _ in range(traj_num)]
    trajs_g_list = [np.array([]) for _ in range(traj_num)]
    parts = []
    for frame_idx, (trajs_pred, trajs_g) in enumerate(zip(batch_trajs_pred, batch_trajs_g)):
        for traj_idx, (pos_pred, pos_g) in enumerate(zip(trajs_pred, trajs_g)):
            if (pos_g != np.array([0., 0.])).all() and (pos_pred != np.array([0., 0.])).all():
                if traj_idx not in parts:
                    parts.append(traj_idx)
                    trajs_pred_list[int(traj_idx)] = np.append(trajs_pred_list[int(traj_idx)], pos_g)   
                    trajs_g_list[int(traj_idx)] = np.append(trajs_g_list[int(traj_idx)], pos_g)                                 
                trajs_pred_list[int(traj_idx)] = np.append(trajs_pred_list[int(traj_idx)], pos_pred)
                trajs_g_list[int(traj_idx)] = np.append(trajs_g_list[int(traj_idx)], pos_g)


    # prev_trajs_np = prev_trajs.cpu().data.numpy()
    # prev_trajs_list = [np.array([]) for _ in range(traj_num)]
    # for frame_idx, p_trajs in enumerate(prev_trajs_np):
    #     for traj_idx, pos in enumerate(p_trajs):
    #         if (pos != np.array([0., 0.])).all():
    #             prev_trajs_list[int(traj_idx)] = np.append(prev_trajs_list[int(traj_idx)], pos)  

    total_trajs_np = total_trajs.cpu().data.numpy()
    total_trajs_list = [np.array([]) for _ in range(traj_num)]
    for frame_idx, p_trajs in enumerate(total_trajs_np):
        for traj_idx, pos in enumerate(p_trajs):
            if (pos != np.array([0., 0.])).all():
                total_trajs_list[int(traj_idx)] = np.append(total_trajs_list[int(traj_idx)], pos)

    plt.figure()
    for plot_idx, traj_idx in enumerate(parts):
        print(f"plotting {traj_idx}")
        total_x = total_trajs_list[traj_idx][::2]
        total_y = total_trajs_list[traj_idx][1::2]
        plt.plot(total_x, total_y, linestyle="dashed", label="total"+str(traj_idx), marker=".")
        for i, (x, y) in enumerate(zip(total_x, total_y)):
            if i < len(total_x)-1:
                plt.arrow(x, y, (total_x[i+1] - x)/2, (total_y[i+1] - y)/2, head_width=0.03, head_length=0.07)
        pred_x = trajs_pred_list[traj_idx][::2]
        pred_y = trajs_pred_list[traj_idx][1::2]            
        plt.plot(pred_x, pred_y, label="pred"+str(traj_idx), marker=".")
        for i, (x, y) in enumerate(zip(pred_x, pred_y)):
            if i < len(pred_x)-1:
                plt.arrow(x, y, (pred_x[i+1] - x)/2, (pred_y[i+1] - y)/2, head_width=0.03, head_length=0.07)        
        # plt.plot(trajs_g_list[traj_idx][::2], trajs_g_list[traj_idx][1::2], linestyle="dotted", label="g"+str(traj_idx), marker=".")
        # plt.plot(prev_trajs_list[traj_idx][::2], prev_trajs_list[traj_idx][1::2], linestyle="dotted", label="prev"+str(traj_idx), marker=".")

        ax = plt.gca()
        #set limits
        # ax.set_xlim([4,13.5])        
        for i in range(2*plot_idx, 2*plot_idx+2):
            line = ax.lines[i]
            print(f"{line.get_label()}\n{line.get_xydata()}\n-------------------------")
        print("===============================")

        plt.legend(loc="upper right")
        plt.title(f"batch {batch_idx}")
    plt.savefig("letssee.png")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("using", device)
    # sl = train(12, 20)
    sl1 = torch.load("sl_4_1.pt")
    validate(sl1, 12, 20)
