#!/usr/bin/env zsh

git push p master

ssh pi@192.168.137.231 "cd /home/pi/nuedc ; git pull p master"
ssh pi@192.168.137.232 "cd /home/pi/nuedc ; git pull p master"