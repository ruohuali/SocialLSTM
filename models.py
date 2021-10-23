from main import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
class SocialLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=20, mediate_dim=128, output_dim=2, social_dim=16, traj_num=3, dropout_prob=0.0,
                N_size=2, grid_cell_size=0.3):
        super(SocialLSTM, self).__init__()
        #specify params
        self.input_dim, self.mediate_dim, self.output_dim, self.hidden_dim = input_dim, mediate_dim, output_dim, hidden_dim
        self.traj_num = traj_num
        self.grid_cell_size = grid_cell_size
        self.N_size = N_size if N_size % 2 == 0 else N_size + 1      
        #specify embedding layers
        self.InputEmbedding = nn.Linear(input_dim, mediate_dim)
        self.SocialEmbedding = nn.Linear((self.N_size+1)**2*self.hidden_dim, social_dim)        
        self.LSTMCell = nn.LSTMCell(mediate_dim+social_dim, hidden_dim)        
        self.OutputLayer = nn.Linear(hidden_dim, output_dim)
        self.Phi = Phi(dropout_prob=dropout_prob)
        self.CorrNormLayer = nn.Sigmoid()

    
    def socialPooling(self, h_tm1, coords, mask):
        with torch.no_grad():
            H = torch.zeros(coords.shape[0], self.N_size+1, self.N_size+1, self.hidden_dim, device=device)
            #calc margin points
            margin_thick = 2*self.N_size*self.grid_cell_size
            leftmost = torch.min(coords[:,0])-margin_thick
            rightmost = torch.max(coords[:,0])+margin_thick
            topmost = torch.min(coords[:,1])-margin_thick
            bottommost = torch.max(coords[:,1])+margin_thick
            ltcorner = torch.tensor([leftmost, topmost], device=device)

            #calc global grid coords
            POS = [[int(xoy) for xoy in (coords[traj_idx]-ltcorner)//self.grid_cell_size]
                    if mask[traj_idx] != 0 else [0,0] for traj_idx in range(coords.shape[0])]
            h_tm1_masked = mask.clone().view(mask.shape[0],1).expand(mask.shape[0],self.hidden_dim) * h_tm1.clone()

            #calc global htm1 matrix
            GRID_width, GRID_height = int((rightmost-leftmost)//self.grid_cell_size), int((bottommost-topmost)//self.grid_cell_size)
            GRID_htm1 = torch.zeros(GRID_width,GRID_height,self.hidden_dim,device=device)
            for traj_idx in range(coords.shape[0]):
                GRID_htm1[POS[traj_idx][0]][POS[traj_idx][1]] += h_tm1[traj_idx]

            #calc H
            for traj_idx in range(coords.shape[0]):
                if mask[traj_idx] != 0:
                    x, y = POS[traj_idx][0], POS[traj_idx][1]
                    R = self.grid_cell_size*self.N_size/2
                    fuck = GRID_htm1[int(x-R):int(x+R),int(y-R):int(y+R),:]
                    H[traj_idx] = GRID_htm1[int(x-R):int(x+R),int(y-R):int(y+R),:]

            H = H.reshape(coords.shape[0], (self.N_size+1)**2*self.hidden_dim)
        return H    


    def forward(self, X, coords, part_masks, all_h_t, all_c_t, Y, T_obs, T_pred):
        outputs = torch.empty(X.shape[0], X.shape[1], self.output_dim, device=device)
        #array of abs coords
        #get the splitting points after which pred starts        
        last_points = coords[T_obs+1,:]     
        
        for frame_idx, (x, coord) in enumerate(zip(X, coords)): 
            if X.shape[1] > 50:    
                print(f"f   {frame_idx}", end='\r') 
            if frame_idx > T_pred: 
                outputs[frame_idx] = torch.zeros(X.shape[1], self.output_dim)
                continue
                
            elif frame_idx <= T_obs:    
                #calc input embedding
                r = self.Phi(self.InputEmbedding(x))
                #calc social pooling embedding
                H = self.socialPooling(all_h_t, coord, part_masks[frame_idx][0])
                e = self.Phi(self.SocialEmbedding(H))
                concat_embed = torch.cat((r,e), 1)
                all_h_t, all_c_t = self.LSTMCell(concat_embed, (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
                
            elif frame_idx <= T_pred and frame_idx > T_obs:                
                #get the abs coords of each traj according to the last points
                last_offs = outputs[frame_idx-1].clone()
                for traj_idx in range(last_points.shape[0]):
                    last_points[traj_idx] += last_offs[traj_idx]
                last_points += last_offs
                #calc input embedding
                r = self.Phi(self.InputEmbedding(last_offs))
                #calc social pooling embedding
                H = self.socialPooling(all_h_t, last_points, part_masks[frame_idx][0])
                e = self.Phi(self.SocialEmbedding(H))
                concat_embed = torch.cat((r,e), 1)                
                all_h_t, all_c_t = self.LSTMCell(concat_embed, (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask                
                

            #dirty fix for appearance that's too short
            if frame_idx > 3 and frame_idx > T_obs:
                for traj_idx in torch.where(part_masks[frame_idx][0] != 0)[0]:
                    if part_masks[frame_idx-3][0][traj_idx] == 0:
                        outputs[frame_idx, traj_idx] = Y[frame_idx, traj_idx] 

        return outputs    


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

    def forward(self, X, part_masks, all_h_t, all_c_t, Y, T_obs, T_pred):
        outputs = torch.empty(X.shape[0], X.shape[1], self.output_dim, device=device)
        for frame_idx, x in enumerate(X):      
            if frame_idx > T_pred:
                outputs[frame_idx] = torch.zeros(X.shape[1], self.output_dim)
                continue
                
            elif frame_idx <= T_obs:
                r = self.Phi(self.InputLayer(x))
                all_h_t, all_c_t = self.LSTMCell(r, (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask
                
            elif frame_idx > T_obs and frame_idx <= T_pred:
                r = self.Phi(self.InputLayer(outputs[frame_idx-1].clone()))
                all_h_t, all_c_t = self.LSTMCell(r, (all_h_t, all_c_t))
                part_mask = torch.t(part_masks[frame_idx]).expand(part_masks[frame_idx].shape[1], self.output_dim)
                outputs[frame_idx] = self.OutputLayer(all_h_t) * part_mask                
                
            #dirty fix for appearance that's too short
            if frame_idx > 3 and frame_idx > T_obs:
                for traj_idx in torch.where(part_masks[frame_idx][0] != 0)[0]:
                    if part_masks[frame_idx-3][0][traj_idx] == 0:
                        outputs[frame_idx, traj_idx] = Y[frame_idx, traj_idx] 

        return outputs
