#!/bin/bash

# --- 設定顏色 ---
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- 函數：錯誤處理 ---
handle_error() {
    echo -e "${RED}錯誤：$1${NC}"
    exit 1
}

# --- 步驟 1：檢查並安裝 Docker ---
if ! command -v docker &> /dev/null; then
    echo -e "${GREEN}正在安裝 Docker...${NC}"
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce
    sudo usermod -aG docker ${USER}
    echo -e "${GREEN}Docker 安裝完成。請重新登入以使群組變更生效。${NC}"
else
    echo "Docker 已安裝。"
fi

# --- 步驟 2：檢查並安裝 Docker Compose ---
if ! command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}正在安裝 Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose 安裝完成。${NC}"
else
    echo "Docker Compose 已安裝。"
fi

# --- 步驟 3：從 Git 克隆或更新專案 ---
if [ -d ".git" ]; then
    echo -e "${GREEN}正在更新專案...${NC}"
    git pull origin main || handle_error "無法更新專案。"
else
    echo -e "${GREEN}正在克隆專案...${NC}"
    git clone https://github.com/yangtandev/face_system.git . || handle_error "無法克隆專案。"
fi

# --- 步驟 4：構建並啟動 Docker 容器 ---
echo -e "${GREEN}正在構建並啟動 Docker 容器...${NC}"
docker-compose up --build -d || handle_error "無法啟動 Docker 容器。"

echo -e "${GREEN}專案已成功啟動！${NC}"
