# --- 基礎映像檔 ---
# 使用 Ubuntu 22.04 作為基礎，這是一個穩定且廣泛使用的版本
FROM ubuntu:22.04

# --- 設定環境變數 ---
# 設定為非互動模式，避免在安裝過程中跳出詢問視窗
ENV DEBIAN_FRONTEND=noninteractive
# 讓 Python 的輸出直接顯示在終端機，方便 docker logs 查看
ENV PYTHONUNBUFFERED=1
# 設定語言環境為 C.UTF-8，確保系統支援 Unicode
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# --- 安裝系統依賴 ---
# 更新套件列表並安裝 Python、pip 以及運行 Qt GUI 應用程式所需的完整函式庫
RUN apt-get update && apt-get install -y \
    # 中文字型 (解決亂碼問題)
    fonts-wqy-zenhei \
    # 基礎工具
    git \
    python3.10 \
    python3-pip \
    python3-tk \
    # 基礎 GUI 函式庫
    libgl1-mesa-glx \
    libglib2.0-0 \
    # 專門為 Qt on XCB 插件補全的 X11 函式庫 (非常重要)
    libx11-xcb1 \
    libxcb1 \
    libxcb-glx0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    # Qt5 核心函式庫 (如果你的 Python 套件依賴 Qt5)
    libqt5widgets5 \
    libqt5gui5 \
    libqt5dbus5 \
    # 清理 apt 快取，縮小映像檔體積
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- 設定工作目錄 ---
# 在容器內建立一個 /app 資料夾，並將其設為工作目錄
WORKDIR /app

# --- 複製專案檔案 ---
# 將目前資料夾（除了 .dockerignore 中指定的檔案）的所有內容複製到容器的 /app 目錄
COPY . .

# --- 安裝 Python 依賴 ---
# 先從 GitHub 下載 YOLOv10，然後安裝它，接著再安裝 requirements.txt 中的其他套件
# 這樣可以確保 YOLOv10 及其包含的 ultralytics 模組被正確安裝
RUN git clone https://github.com/THU-MIG/yolov10.git && \
    pip3 install --no-cache-dir ./yolov10 && \
    pip3 install --no-cache-dir -r requirements.txt

# --- 設定啟動指令 ---
# 設定容器啟動時要執行的預設指令
CMD ["python3", "main.py"]
