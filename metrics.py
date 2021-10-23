from main import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
def calcCoordinatesNew(start_point, offsets):
    next_point = start_point
    coords = torch.zeros(offsets.shape[0]+1,offsets.shape[1],offsets.shape[2])
    for t, offset in enumerate(offsets):
        next_point += offset
        coords[t] = next_point.clone()
    coords = coords.reshape((40,2))
    return coords

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
    #print(f"avg disp error {result}")
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
    #print(f"final disp error {result}")
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
    plt.savefig("eth_plots/"+str(batch_idx)+str(is_total))    

