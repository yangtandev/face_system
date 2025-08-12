#!/bin/bash

function taskfun () {
    echo "使用者gini-face存在, 正在檢查並設定crontab任務..."

    # 檢查任務是否已存在
    CURRENT_CRON=$(crontab -u gini-face -l 2>/dev/null)
    TASK="$2"

    if echo "$CURRENT_CRON" | grep -qF "$TASK"; then
        echo "crontab任務已存在, 跳過設定。"
    else
        # 任務不存在，則添加
        (crontab -u gini-face -l 2>/dev/null; echo "$TASK") | crontab -u gini-face -
        echo "crontab任務已成功為使用者gini-face添加 $1。"
    fi
}

# 檢查是否使用root用戶運行
if [ "$(id -u)" -ne 0 ]; then
    echo "請使用root權限運行該腳本。"
    exit 1
fi

# 檢查使用者gini-face是否存在
if id "gini-face" &>/dev/null; then
    #taskfun "run_main" "@reboot sleep 3; bash /home/gini-face/face_system/run_main.sh >> /home/gini-face/open_main.log 2>&1"
    taskfun "update" "*/5  * * * * bash /home/gini-face/face_system/update.sh"
    taskfun "clean_log" "*/5  * * * * bash /home/gini-face/face_system/clean_log.sh"

else
    echo "使用者gini-face不存在, 請檢查使用者名稱。"
    exit 1
fi