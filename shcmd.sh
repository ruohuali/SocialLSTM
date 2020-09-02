#!/bin/bash

#remove old pics
echo "removing old pics"
cd eth_plots
rm **
cd ..

#start training
echo "initiating training"
python3 fuck.py
echo "training is done"

#git push
echo "start pushing"
git add .
git commit -m "auto"
git push
expect -exact "Username for 'https://github.com': "
send "ruohuali"
expect "Password for 'https://ruohuali@github.com': "
send "701218Zhang"
echo "end pushing"
