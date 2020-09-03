#!/bin/bash #!/bin/expect

n = $"\n\n"
user = $"ruohuali"
pass = $"701218Zhang"

#git push
echo "start pushing"
echo "***********"
git add .
git commit -m "auto"
git push
expect "Username for 'https://github.com': "
send "$user"
echo "$n"
echo "end pushing"