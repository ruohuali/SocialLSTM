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
expect "Username for 'https://github.com': "
send "ruohuali\n"
expect "Password for 'https://ruohuali@github.com': "
send "701218Zhang\n"
echo "end pushing"
