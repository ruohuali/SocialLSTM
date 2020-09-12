def plotting_batch(batch_trajs_pred_gpu, input_seq, dataset, T_obs, is_total, batch_idx):          
    #reform the trajs tensor to a list of each traj's pos at each frame
    batch_trajs_pred = batch_trajs_pred_gpu.cpu().data.numpy()
    trajs_pred_list = []
    for traj_idx in batch_trajs_pred.shape[1]:
        trajs_pred_list[traj_idx] = batch_trajs_pred[:,traj_idx,:]    

    trajs_coord_list = []
    for traj_idx in batch_trajs_pred.shape[1]:
        for t in input_seq.shape[0]:
            trajs_coord_list[traj_idx].append(dataset.getCoordinates(input_seq[t,traj_idx,0],input_seq[t,traj_idx,1]))
        trajs_coord_list[traj_idx].append(dataset.getCoordinates(input_seq[t+1,traj_idx,0]+dataset.time_step,input_seq[t,traj_idx,1]))

    #calc the coords of each step
    traj_pred_coords = [[] for _ in range(len(trajs_coord_list))]
    for traj_idx, traj in enumerate(trajs_coord_list):
        split_point = trajs_coord_list[traj_idx][T_obs+1]
        traj_pred_coords[traj_idx].append(split_point)
        next_point = split_point
        for off in traj:
            next_point += off
            traj_pred_coords[traj_idx].append(next_point)

        
    #plot
    plt.figure(figsize=(30,30))
    plt.xlim([-15,15])
    plt.ylim([-15,15])
    plot_idx = 0
    for traj_idx in parts:
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

        total_x = batch_coords[:,traj_idx,2]        
        total_x = total_x[np.where(total_x != 0.)]
        total_y = batch_coords[:,traj_idx,3]
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