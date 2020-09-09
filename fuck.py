from VanillaLSTM import *

def visualizeFile(file, batch_size, name="try"):
    dataset = FramesDataset(file)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch_idx, data in enumerate(dataloader):
        batch_coords = data['coords']
        visualizeBatch(batch_coords.cpu().data)

def visualizeBatch(batch_coords, batch_idx):
    #plot
    plt.figure(figsize=(12,12))
    plt.xlim([-2,15])
    plt.ylim([-2,15])
    plot_idx = 0
    for traj_idx in parts:
        total_x = batch_coords[:,traj_idx,2]        
        total_x = total_x[np.where(total_x != 0.)]
        total_y = batch_coords[:,traj_idx,3]
        total_y = total_y[np.where(total_y != 0.)]       
        try:
            plt.plot(total_x, total_y, linestyle="dashed", label="total"+str(traj_idx))
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
    plt.savefig("eth_plots/"+str(batch_idx)+"vv")

if __name__ == "__main__":
    files_dir = "try_dataset"
    files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]

