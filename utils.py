from main import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    def preprocessBatch(self, file_data_in):
        file_data = sorted(file_data_in, key=lambda data : data[1])
        file_data_sort = sorted(file_data_in, key=lambda data : data[0])
        
        #turn the file into time-major multidimensional tensor
        traj_list, participant_masks, coord_tensors = self.text2Tensor(file_data_sort)
        
        #process the file data such that it contains the offsets not global coords
        file_data_off = []
        for i, line in enumerate(file_data):
            if i > 0:
                if file_data[i][1] == file_data[i-1][1]:
                    file_data_off.append([file_data[i-1][0], file_data[i-1][1], file_data[i][2]-file_data[i-1][2], file_data[i][3]-file_data[i-1][3]])
        file_data_off.sort(key=lambda data : data[0])        
        
        traj_list, participant_masks, off_tensors = self.text2Tensor(file_data_off)


        # #get offsets
        # off_tensors = torch.zeros(coord_tensors.shape[0]-1,coord_tensors.shape[1],coord_tensors.shape[2])
        # for t in range(coord_tensors.shape[0]-1):
        #     off_tensors[t] = coord_tensors[t+1] - coord_tensors[t]

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
        self.special = True if ".p" in path else False
        if not self.special:
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


def printPics(v,j):
    plt.figure()
    plt.title(str(j))
    for i, data in enumerate(v):
        if len(data) != 0:
            plt.plot(np.arange(len(v[0])), data)        
    plt.savefig("eth_plots/"+"train"+str(j))
    print("--->","eth_plots/"+"train"+str(j))
