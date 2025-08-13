#!/bin/bash

# 执行git pull命令，并将结果保存到变量中
cd /home/$LOGNAME/face_system
#git config pull.rebase false
pull_result=$(git pull origin main)

# 检查结果中是否包含 "Updating" 或 "Fast-forward" 字样
if [[ $pull_result = *"Updating"* || $pull_result = *"Fast-forward"* ]]; then
    bash ./restart.sh
else
    echo "Repository is already up to date."
fi
###