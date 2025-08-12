#!/bin/bash

function taskfun () {
    echo "使用者$1 存在, 正在檢查並設定crontab任務..."

    # 檢查任務是否已存在
    CURRENT_CRON=$(crontab -u $1 -l 2>/dev/null)
    TASK="$3"

    if echo "$CURRENT_CRON" | grep -qF "$TASK"; then
        echo "crontab任務已存在, 跳過設定。"
    else
        # 任務不存在，則添加
        (crontab -u $1 -l 2>/dev/null; echo "$TASK") | crontab -u $1 -
        echo "crontab任務已成功為使用者$1 添加 $2。"
    fi
}

# 更新套件列表

sudo apt-get update -y

sudo apt install openssh-server -y

sudo apt-get install libxcb-xinerama0 -y

sudo apt-get install git-all -y

sudo apt install openvswitch-switch -y

if ! command -v python3 &> /dev/null; then
    echo "Python 3 未安裝。正在安裝 Python 3.10.12..."

    # 安裝所需的依賴
    sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

    # 下載 Python 3.10.12 的原始碼
    wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz

    # 解壓縮
    tar -xf Python-3.10.12.tgz

    # 進入解壓縮後的目錄
    cd Python-3.10.12

    # 配置安裝選項
    ./configure --enable-87

    # 編譯並安裝
    make -j $(nproc)
    sudo make altinstall

    # 返回原始目錄
    cd ..

    # 清理安裝檔案
    rm -rf Python-3.10.12 Python-3.10.12.tgz

    echo "Python 3.10.12 已安裝完畢。"

    sudo apt-get install python3-pip
else
    echo "已安裝 Python 3。"
fi

#sudo apt-get install python3-pip

# 檢查是否已安裝 python3.10-venv
if ! dpkg -s python3.10-venv &> /dev/null; then
    echo "python3.10-venv 未安裝。正在安裝 python3.10-venv..."

    # 安裝 python3.10-venv
    sudo apt-get install -y python3.10-venv

    echo "python3.10-venv 已安裝完畢。"
else
    echo "python3.10-venv 已安裝。"
fi


# 獲取當前使用者名稱
USER_NAME="$1"

# 檢查使用者gini-face是否存在
if id "$USER_NAME" &>/dev/null; then
    #taskfun $USER_NAME "run_main" "@reboot sleep 3; bash /home/$USER_NAME/face_system/run_main.sh >> /home/$USER_NAME/open_main.log 2>&1"
    #taskfun $USER_NAME "# update" "*/5  * * * * bash /home/$USER_NAME/face_system/update.sh"
    taskfun $USER_NAME "clean_log" "*/5  * * * * bash /home/$USER_NAME/face_system/clean_log.sh"

else
    echo "使用者 $USER_NAME 不存在, 請檢查使用者名稱。"
    exit 1
fi

# 建立虛擬環境的目錄
VENV_DIR="/home/$USER_NAME/face_system/venv"

# 如果目錄不存在，則建立它
if [ ! -d "$VENV_DIR" ]; then
    mkdir -p "$VENV_DIR"
fi

# 在指定目錄下建立虛擬環境
python3 -m venv "$VENV_DIR"

echo "Python 虛擬環境已在 $VENV_DIR 中建立。"

if [ -d "./site-packages" ]; then
    cp -r ./site-packages /home/$USER_NAME/new_face_Sys/lib/python3.10/
    echo "複製python套件"
else
    source "/home/$USER_NAME/new_face_Sys/bin/activate"

    cd "/home/$USER_NAME/face_system"

    pip3 install -r "/home/$USER_NAME/face_system/requirements.txt"
fi

echo "設定ssh"

sudo chmod 700 ./.ssh
sudo chmod 700 ./.ssh/id_rsa
sudo chmod 700 ./.ssh/id_rsa.pub
sudo cp -r ./.ssh /home/$USER_NAME/
sudo chmod 777 /home/set_.log
sudo cp -r ./.ssh /home/root/

echo "移動相關檔案"
#cp -r /home/face_system /home/$USER_NAME/
#cp -r ./20180402-114759-vggface2.pt /home/$USER_NAME
cd /home/$USER_NAME/
sudo -u $USER_NAME  git clone git@github.com:Hao-Liang233/face_system.git
#mv /home/$USER_NAME/20180402-114759-vggface2.pt  /home/$USER_NAME/face_system/models/data/20180402-114759-vggface2.pt

sudo chmod 775 /home/$USER_NAME/face_system -R
sudo chmod 775 /home/$USER_NAME/new_face_Sys -R
sudo chown $USER_NAME /home/$USER_NAME/face_system -R
sudo chown $USER_NAME /home/$USER_NAME/new_face_Sys -R

echo "建立捷徑"
# 定義 Desktop Entry 文件的內容
DESKTOP_ENTRY="[Desktop Entry]
Encoding=UTF-8
Name=face_app
Exec=bash /home/$USER_NAME/face_system/run_main.sh
Terminal=true
Type=Application
StartupNotify=true"

# 在用戶主目錄下創建 new_face_sys.desktop 文件
echo "$DESKTOP_ENTRY" > /home/$USER_NAME/new_face_sys.desktop
sudo chmod 644 /home/$USER_NAME/new_face_sys.desktop
# 移動 new_face_sys.desktop 到 /usr/share/applications 資料夾
sudo cp /home/$USER_NAME/new_face_sys.desktop /usr/share/applications/
sudo cp /home/$USER_NAME/new_face_sys.desktop /home/$USER_NAME/.config/autostart/
sudo cp /home/$USER_NAME/new_face_sys.desktop /home/$USER_NAME/.config/autostart/
# 設置正確的權限
sudo chmod 644 /usr/share/applications/new_face_sys.desktop

echo "結束"
