#!/bin/bash #!/bin/expect

n = $"***********************************************\n\n"
#remove old pics
echo "removing old pics"
echo "$n"
rm -rf eth_plots
mkdir eth_plots
echo "$n"

#start training
echo "initiating training"
echo "$n"
python3 VanillaLSTM.py
echo "$n"
echo "training is done"

#git push
echo "start pushing"
echo "$n"
git add .
git commit -m "auto"
git push
echo "$n"
echo "end pushing"
