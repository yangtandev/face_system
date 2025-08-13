#!/bin/bash

# --- 說明 ---
# 這個腳本會自動化 Docker 容器的建構與啟動流程。
# 它會先設定 X Server 的存取權限，然後使用 docker-compose 來啟動應用程式。
# 腳本中包含錯誤檢查，以確保每一步都成功執行。

# 設定顏色，讓輸出更易讀
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- 步驟 1: 允許本地使用者存取 X Server ---
# 這個指令是讓 Docker 容器能夠將 GUI 顯示在主機螢幕上的關鍵。
echo -e "${GREEN}正在設定 X Server 存取權限...${NC}"
xhost +local:
# 檢查上一條指令是否成功執行
if [ $? -ne 0 ]; then
    echo -e "${RED}錯誤：設定 xhost 失敗。請確認您在圖形介面環境下執行此腳本，且 xhost 指令已安裝。${NC}"
    exit 1
fi
echo "權限設定完成。"
echo ""

# --- 步驟 2: 使用 Docker Compose 建構並啟動容器 ---
# --build: 強制重新建構映像檔，以確保使用最新的程式碼。
# -d:      在背景 (detached) 模式下執行。
echo -e "${GREEN}正在使用 Docker Compose 建構並啟動容器...${NC}"
echo "這可能需要幾分鐘，特別是第一次建構時。"
docker compose up --build -d

# 檢查 docker-compose 是否成功啟動
if [ $? -ne 0 ]; then
    echo -e "${RED}錯誤：docker-compose 啟動失敗。請檢查 Docker 是否正在運行，以及 docker-compose.yml 檔案是否存在且內容正確。${NC}"
    exit 1
fi

echo ""
echo "--------------------------------------------------"
echo -e "${GREEN}Docker 容器已成功啟動！${NC}"
echo "您的應用程式應該很快就會顯示在螢幕上。"
echo ""
echo "您可以使用以下指令來查看日誌："
echo "  docker compose logs -f"
echo ""
echo "若要停止應用程式，請執行："
echo "  docker compose down"
echo "--------------------------------------------------"

exit 0
