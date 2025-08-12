#!/bin/bash

# 檢查是否有正在運行的 faceNet_with_qt.py 或 main.py
faceNet_process=$(ps aux | grep "faceNet_with_qt.py" | grep -v "grep")
main_process=$(ps aux | grep "main.py" | grep -v "grep")

# 如果發現 faceNet_with_qt.py 正在運行
if [ -n "$faceNet_process" ]; then
    echo "發現正在運行的 faceNet_with_qt.py 進程，正在關閉..."
    # 使用 pkill 關閉所有與 faceNet_with_qt.py 有關的進程
    pkill -f "faceNet_with_qt.py"
    echo "faceNet_with_qt.py 已關閉。"
fi

# 如果發現 main.py 正在運行
if [ -n "$main_process" ]; then
    echo "發現正在運行的 main.py 進程，正在關閉..."
    # 使用 pkill 關閉所有與 main.py 有關的進程
    pkill -f "main.py"
    echo "main.py 已關閉。"
fi

# 重新啟動 bash /home/$LOGNAME/face_system/run_main.sh
echo "正在重新啟動 face system..."
bash /home/$LOGNAME/face_system/run_main.sh >> /home/$LOGNAME/open_main.log 2>&1

echo "face system 已重新啟動。"
