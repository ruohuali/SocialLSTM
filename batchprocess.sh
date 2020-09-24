#!/bin/bash

echo "starting"

for i in {100..156}
do
   echo "$i * 550"
   python3 main.py 'v' --special_model 'a_just_trained_model_for_eth.pt' --special_file 'x_all.p' --special_start $2 --T_obs 20 --T_pred 40
done
