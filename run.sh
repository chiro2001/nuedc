#!/usr/bin/env zsh
cd /home/pi/nuedc
export PYTHONUNBUFFERED=1
export DISPLAY=:0.0
export MVCAM_SDK_PATH=/opt/MvCamCtrlSDK
export LD_LIBRARY_PATH=/opt/MvCamCtrlSDK/lib/armhf
export MVCAM_COMMON_RUNENV=/opt/MvCamCtrlSDK/lib
echo $$ > run.pid
python3 main.py

