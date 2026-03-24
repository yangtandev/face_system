#!/bin/bash
set -e

# --- 1. Sudo 提權與 Keep-alive ---
echo "▶ 為了安裝系統套件與設定服務，需要取得 sudo 權限。"
sudo -v
# Keep-alive in background
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# --- 2. 變數與路徑判定 ---
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CURRENT_USER="$USER"

cd "$PROJECT_DIR"

echo ""
echo "============================================="
echo "   Face System 系統環境全自動安裝精靈"
echo "============================================="
echo ""

# --- 3. 收集使用者輸入參數 ---
read -p "請輸入伺服器 IP [預設: 43.213.128.240]: " INPUT_SERVER_IP
SERVER_IP=${INPUT_SERVER_IP:-43.213.128.240}

read -p "請輸入伺服器使用者名稱 [預設: ubuntu]: " INPUT_SERVER_USER
SERVER_USER=${INPUT_SERVER_USER:-ubuntu}

read -p "請輸入 IN Camera RTSP 網址 [預設: rtsp://127.0.0.1/in]: " INPUT_IN_CAMERA
IN_CAMERA=${INPUT_IN_CAMERA:-"rtsp://127.0.0.1/in"}

read -p "請輸入 OUT Camera RTSP 網址 [預設: rtsp://127.0.0.1/out]: " INPUT_OUT_CAMERA
OUT_CAMERA=${INPUT_OUT_CAMERA:-"rtsp://127.0.0.1/out"}

echo ""
echo "▶ 以下為您的設定："
echo "  - 專案路徑: $PROJECT_DIR"
echo "  - 執行身份: $CURRENT_USER"
echo "  - 伺服器 IP: $SERVER_IP"
echo "  - 伺服器使用者: $SERVER_USER"
echo "  - 進入攝影機 (IN): $IN_CAMERA"
echo "  - 離開攝影機 (OUT): $OUT_CAMERA"
echo ""
read -p "請按 Enter 鍵繼續安裝，或按 Ctrl+C 取消..."

# --- 4. 安裝系統依賴套件 ---
echo "▶ [1/6] 安裝基礎與系統依賴套件 (apt-get)..."
sudo apt-get update
sudo apt-get install -y build-essential curl wget git git-lfs software-properties-common \
    python3 python3-pip python3-venv \
    ffmpeg mosquitto mosquitto-clients \
    libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0

# 檢查系統預設 Python3 是否滿足 >= 3.10 (針對 Ubuntu 20.04 等舊系統的防護)
PYTHON_CMD="python3"
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "▶ 警告：系統預設 Python3 版本低於 3.10，將自動為其安裝 Python 3.10 (deadsnakes PPA)..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
    PYTHON_CMD="python3.10"
fi

echo "▶ 啟動並設定 Mosquitto 服務..."
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Git LFS
echo "▶ 拉取大型模型檔案 (git lfs pull)..."
git lfs install
git lfs pull

# --- 5. 建立 Python 虛擬環境與安裝相依套件 ---
echo "▶ [2/6] 建立 Python 虛擬環境 (venv)..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# --- 6. SSH 金鑰設定 ---
echo "▶ [3/6] SSH 金鑰設定..."
SSH_KEY_PATH="/home/$CURRENT_USER/.ssh/id_rsa"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "  >> 未找到 SSH 金鑰，自動產生..."
    ssh-keygen -t rsa -b 4096 -N "" -f "$SSH_KEY_PATH" -C "auto-generated@face-system"
else
    echo "  >> SSH 金鑰已存在 ($SSH_KEY_PATH)"
fi

echo "  >> 接下來將執行 ssh-copy-id 將公鑰傳送至 $SERVER_USER@$SERVER_IP"
echo "  >> (可能需要您手動輸入一次遠端伺服器的密碼以完成認證)"
ssh-copy-id -i "${SSH_KEY_PATH}.pub" "$SERVER_USER@$SERVER_IP" || {
    echo "  ⚠️ ssh-copy-id 未完全成功，可能是網路或密碼錯誤。設定檔仍會產生，稍後可手動檢查。"
}

# --- 7. 更新 config 設定檔 ---
echo "▶ [4/6] 產生 config.json..."
# 使用 Python 內嵌腳本完美修改 config 設定
python - <<EOF
import json
import os
import sys

# 把專案目錄加進 Path 確保找到 setting 目錄
sys.path.append("$PROJECT_DIR")

try:
    from setting.build_config import default_config
except ImportError:
    print("無法載入 setting/build_config.py")
    sys.exit(1)

# 修改記憶體中的 default_config
default_config["Server"]["ip"] = "$SERVER_IP"
default_config["Server"]["username"] = "$SERVER_USER"
default_config["Server"]["ssh_key_path"] = "$SSH_KEY_PATH"
default_config["cameraIP"]["in_camera"] = "$IN_CAMERA"
default_config["cameraIP"]["out_camera"] = "$OUT_CAMERA"

config_path = os.path.join("$PROJECT_DIR", "config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(default_config, f, ensure_ascii=False, indent=2)

print("  >> config.json 已成功寫入！")
EOF

# --- 8. Web 介面 Sudo 免密權限 ---
echo "▶ [5/6] 設定 Web 介面重啟服務的免密碼 sudo 權限..."
SUDOERS_FILE="/etc/sudoers.d/face_system_web_restart"
echo "$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart face_system.service" | sudo tee "$SUDOERS_FILE" > /dev/null
sudo chmod 440 "$SUDOERS_FILE"

# --- 9. Systemd 服務註冊 ---
echo "▶ [6/6] 設定 Systemd 服務..."
SERVICE_FILE_PATH="/etc/systemd/system/face_system.service"

# 確認當前使用者的 XDG_RUNTIME_DIR ID (通常是 1000)
USER_ID=$(id -u "$CURRENT_USER")

sudo tee "$SERVICE_FILE_PATH" > /dev/null <<EOF
[Unit]
Description=Face Recognition System GUI Application
After=graphical.target mosquitto.service

[Service]
User=$CURRENT_USER
Environment=DISPLAY=:0
Environment="XDG_RUNTIME_DIR=/run/user/$USER_ID"
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python -u $PROJECT_DIR/face_main.py
Restart=always
RestartSec=10

[Install]
WantedBy=graphical.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable face_system.service
# 立即啟動服務
sudo systemctl start face_system.service || true

# --- 10. Cronjob 排程設定 ---
echo "▶ 設定自動報表 (analytics_reporter.py) 排程..."
# 確保 log 目錄存在
mkdir -p "$PROJECT_DIR/log"
CRON_JOB="0 * * * * $PROJECT_DIR/venv/bin/python $PROJECT_DIR/analytics_reporter.py >> $PROJECT_DIR/log/reporter.log 2>&1"
# 檢查是否已經存在
(crontab -l 2>/dev/null | grep -v "analytics_reporter.py"; echo "$CRON_JOB") | crontab -
echo "  >> 每小時產出報表之 Cron Job 已生效。"

echo ""
echo "============================================="
echo " 🎉 安裝已全部完成！"
echo "============================================="
echo "您可以透過以下指令查看服務狀態："
echo "  sudo systemctl status face_system.service"
echo "============================================="
