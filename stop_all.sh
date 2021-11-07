#!/usr/bin/env zsh

ssh 192.168.137.231 "sh /home/pi/nuedc/stop.sh" &
ssh 192.168.137.232 "sh /home/pi/nuedc/stop.sh" &
sh stop.sh

