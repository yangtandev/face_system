#!/bin/bash
sleep 5
source "/home/$LOGNAME/face_system/venv/bin/activate"
#xhost +
#export DISPLAY=:0
python3 "/home/$LOGNAME/face_system/face_main.py"
