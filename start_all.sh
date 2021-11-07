#!/usr/bin/env zsh

cd /home/pi/nuedc
#sleep 1
ssh 192.168.137.232 "sh /home/pi/nuedc/run.sh" &
#sleep 1
ssh 192.168.137.231 "sh /home/pi/nuedc/run.sh" &
#fg %1

./run.sh

ssh 192.168.137.231 "sh /home/pi/nuedc/stop.sh"
ssh 192.168.137.232 "sh /home/pi/nuedc/stop.sh"


