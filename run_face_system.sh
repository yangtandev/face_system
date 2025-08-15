#!/bin/bash
sleep 5
source "/app/venv/bin/activate"
#xhost +
#export DISPLAY=:0
python3 "/app/face_main.py"
