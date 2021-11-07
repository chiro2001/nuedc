#!/usr/bin/env zsh

cd /home/pi/nuedc
./run.sh & echo $! > run.pid
sleep 1
ssh 192.168.137.232 "cd /home/pi/nuedc; ./run.sh & echo $! > run.pid" &
sleep 1
ssh 192.168.137.231 "cd /home/pi/nuedc; ./run.sh & echo $! > run.pid" &

fg

ssh 192.168.137.231 "cd /home/pi/nuedc; kill -9 `cat run.pid`"
ssh 192.168.137.232 "cd /home/pi/nuedc; kill -9 `cat run.pid`"


