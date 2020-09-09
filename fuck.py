from VanillaLSTM import *

def visualizeFile(file, T_obs, T_pred):
    print(f"file {file}")
    dataset = FramesDataset(file)
    dataloader = DataLoader(dataset, batch_size=T_pred)
    for batch_idx, data in enumerate(dataloader):
        print(f"batch {batch}")
        batch_coords = data['coords']
        visualizeBatch(batch_coords.cpu().data.numpy(), batch_idx, T_obs, name=file.replace(".txt",""))

def visualizeBatch(batch_coords, batch_idx, T_obs, name="name"):
    #plot
    plt.figure(figsize=(12,12))
    plt.xlim([-2,15])
    plt.ylim([-2,15])
    plot_idx = 0
    for traj_idx in range(batch_coords.shape[1]):
        total_x = batch_coords[:,traj_idx,2]        
        total_x = total_x[np.where(total_x != 0.)]
        total_y = batch_coords[:,traj_idx,3]
        total_y = total_y[np.where(total_y != 0.)]
        obs_x, obs_y = batch_coords[T_obs,traj_idx,2], batch_coords[T_obs,traj_idx,3]  
        try:
            plt.plot(total_x, total_y, linestyle="dashed", label="total"+str(traj_idx))
            plt.plot(obs_x, obs_y, label="total"+str(traj_idx), marker='*')
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
    plt.title(f"{name} batch {batch_idx}")
    plt.savefig(name+str(batch_idx)+"vv")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    files_dir = "try_dataset"
    files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
    for file in files:
        visualizeFile(file, 8, 20)
