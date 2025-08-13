# --- 基礎映像檔 ---
# 使用 Ubuntu 22.04 作為基礎，這是一個穩定且廣泛使用的版本
FROM ubuntu:22.04

# --- 設定環境變數 ---
# 設定為非互動模式，避免在安裝過程中跳出詢問視窗
ENV DEBIAN_FRONTEND=noninteractive
# 讓 Python 的輸出直接顯示在終端機，方便 docker logs 查看
ENV PYTHONUNBUFFERED=1

# --- 安裝系統依賴 ---
# 更新套件列表並安裝 Python、pip 以及 GUI 應用程式所需的函式庫
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # 清理 apt 快取，縮小映像檔體積
    && rm -rf /var/lib/apt/lists/*

# --- 設定工作目錄 ---
# 在容器內建立一個 /app 資料夾，並將其設為工作目錄
WORKDIR /app

# --- 複製專案檔案 ---
# 將目前資料夾（除了 .dockerignore 中指定的檔案）的所有內容複製到容器的 /app 目錄
COPY . .

# --- 安裝 Python 依賴 ---
# 使用 pip 安裝 requirements.txt 中定義的所有套件
RUN pip3 install --no-cache-dir -r requirements.txt

# --- 設定啟動指令 ---
# 設定容器啟動時要執行的預設指令
CMD ["python3", "main.py"]
