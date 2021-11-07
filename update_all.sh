#!/usr/bin/env zsh

cd /home/pi/nuedc
git reset --hard && git pull
ssh 192.168.137.231 "cd /home/pi/nuedc ; git reset --hard && git pull" &
ssh 192.168.137.232 "cd /home/pi/nuedc ; git reset --hard && git pull" &
