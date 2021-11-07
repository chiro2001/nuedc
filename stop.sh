#!/usr/bin/env zsh
cd /home/pi/nuedc
kill -9 `ps aux | grep "python3 main.py" | awk '{print $2}' | awk NR==1`
kill -9 `cat run.pid`
