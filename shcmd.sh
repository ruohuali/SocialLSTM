#!/bin/bash #!/bin/expect

#remove old pics
echo "removing old pics"
cd eth_plots
rm **
cd ..

#start training
echo "initiating training"
python3 fuck.py
rm fuck.py
echo "training is done"

#git push
echo "start pushing"
git add .
git commit -m "auto"
git push
echo "end pushing"
