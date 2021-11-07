#!/usr/bin/env zsh

cd /home/pi/nuedc
./run.sh &
ssh 192.168.137.232 "sh /home/pi/nuedc/run.sh" &

ssh 192.168.137.231 "sh /home/pi/nuedc/run.sh" &

echo "Press enter to stop"
read

ssh 192.168.137.231 "sh /home/pi/nuedc/stop.sh" &
ssh 192.168.137.232 "sh /home/pi/nuedc/stop.sh" &
sh stop.sh

