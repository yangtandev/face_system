#!/bin/bash
sleep 10
#export DISPLAY=:0
source "/home/$LOGNAME/face_system/venv/bin/activate"
python3 "/home/$LOGNAME/face_system/main.py"
