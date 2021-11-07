#!/usr/bin/env zsh

cd /home/pi/nuedc
./run.sh & echo $! > run.pid
sleep 1
ssh 192.168.137.232 "cd /home/pi/nuedc; ./run.sh & echo $! > run.pid" &
sleep 1
ssh 192.168.137.231 "cd /home/pi/nuedc; ./run.sh & echo $! > run.pid" &

fg 1

ssh 192.168.137.231 "cd /home/pi/nuedc; ./stop.sh"
ssh 192.168.137.232 "cd /home/pi/nuedc; ./stop.sh"


