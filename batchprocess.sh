#!/bin/bash #!/bin/expect

n = $"***********************************************\n\n"

echo "starting"

for i in {100..156}
do
   echo "$i * 550"
   python3 VanillaLSTM.py $i
done
