#!/bin/bash #!/bin/expect

n = $"***********************************************\n\n"

echo "starting"
echo "$n"
python3 VanillaLSTM.py
echo "$n"
echo "training is done"

for i in {0..3}
do
   echo "$i * 550"
   python3 VanillaLSTM.py $i
done
