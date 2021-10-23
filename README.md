<h2>Implementation of Social LSTM: Human Trajectory Prediction in Crowded Spaces</h2>
<h3>Paper</h3>
<a href="https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf">Social LSTM: Human Trajectory Prediction in Crowded Spaces</a>
<h3>Dependencies</h3>
<ul>
<li>torch 1.6.0</li>
<li>matplotlib 3.2.1</li>
<li>numpy 1.18.4</li>
</ul>
<h3>Files</h3>
There are two models available: SocialLSTM and VanillaLSTM.
<br>
Dataset is located in datasets/[dataset name], where each dataset is a collection of training and validating data.
<br>
Each file in the dataset is of the form 
<br>
<code>frame_number    pedestrian_number   y_coordinates   x_coordinates</code>
<h3>HOWTO</h3>
To train and validate a model against a specific training & validating set, run <br>
<code>python3 main.py mode --dataset [dataset_name] --epoch [epoch_num] --T_obs [observe_step] --T_pred [predict_step]</code><br>
where <kbd>mode</kbd> can be either <kbd>'s'</kbd> or <kbd>'v'</kbd><br>
E.g. to train and validate on "eth" dataset in /datasets, simply run <code>python3 main.py "s" --dataset "eth" --epoch 3 </code> <br>
To only validate a chosen model against a validating set, run <br>
<code>python3 main.py mode --dataset [dataset_name] --pure_val_name [model_dir] --T_obs [observe_step] --T_pred [predict_step]</code><br>
To validate a chosen model against a special validating set, run <br>
<code>python3 main.py mode --special_model [model_dir] --special_file [file_name] --special_start [start_ped] --T_obs [observe_step] --T_pred [predict_step]</code><br>
Special dataset is the dataset of <code>.pkl</code> file with aligned number of frame numbers.
If special dataset is too large to run in one sitting, refer to batchprocess.sh .