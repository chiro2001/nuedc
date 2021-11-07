#!/usr/bin/env zsh
cd /home/pi/nuedc
kill `ps aux | grep "python3 main.py" | awk '{print $2}' | awk NR==1`
#kill `cat run.pid`
