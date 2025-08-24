#!/bin/bash

# --- 設定顏色 ---
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- 函數：錯誤處理 ---
handle_error() {
    echo -e "${RED}錯誤：$1${NC}" >&2
    exit 1
}

# --- 函數：執行命令並檢查結果 ---
run_command() {
    echo -e "${GREEN}>> 執行: $@${NC}"
    "$@"
    if [ $? -ne 0 ]; then
        handle_error "命令 '$*' 執行失敗。"
    fi
}

# --- 步驟 1：檢查並安裝 Docker ---
if ! command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker 未安裝，開始安裝程序...${NC}"
    run_command sudo apt-get update
    run_command sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

    # 添加 Docker 的官方 GPG 金鑰
    sudo install -m 0755 -d /etc/apt/keyrings
    run_command curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # 添加 Docker 的 APT 倉庫
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    run_command sudo apt-get update
    run_command sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # 將當前用戶添加到 docker 組
    run_command sudo usermod -aG docker ${USER}
    echo -e "${GREEN}Docker 及 Docker Compose V2 安裝完成。${NC}"
    echo -e "${RED}重要：請完全登出後再重新登入，或重新啟動你的終端，以使群組權限生效！${NC}"
    # exec newgrp docker
else
    echo "Docker 已安裝。"
    # 檢查 Docker Compose V2 插件是否安裝
    if ! docker compose version &> /dev/null; then
        echo -e "${GREEN}Docker 已安裝，但缺少 Docker Compose V2 插件。正在為您安裝...${NC}"
        run_command sudo apt-get update
        run_command sudo apt-get install -y docker-compose-plugin
        echo -e "${GREEN}Docker Compose V2 插件安裝完成。${NC}"
    fi
fi

# --- 步驟 2：從 Git 克隆或更新專案 ---
if [ ! -d ".git" ]; then
    echo -e "${GREEN}Git 倉庫不存在，請先將專案 clone 下來。${NC}"
    handle_error "請在 face_system 專案目錄下執行此腳本。"
else
    echo -e "${GREEN}正在更新專案...${NC}"
    run_command git pull origin main
fi

# --- 步驟 3：構建並啟動 Docker 容器 (使用 Docker Compose V2) ---
echo -e "${GREEN}正在使用 Docker Compose V2 構建並啟動 Docker 容器...${NC}"
run_command docker compose up --build -d

echo -e "${GREEN}專案已成功啟動！${NC}"
