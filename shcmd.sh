#!/bin/bash #!/bin/expect

#remove old pics
echo "removing old pics"
rm -rf eth_plots
mkdir eth_plots

#start training
echo "initiating training"
python3 VanillaLSTM.py
echo "training is done"

#git push
echo "start pushing"
git add .
git commit -m "auto"
git push
echo "end pushing"
