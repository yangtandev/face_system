# --- 基礎映像檔 ---
FROM ubuntu:22.04

# --- 設定環境變數 ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# 將 yolov10 目錄加入 Python 的模組搜尋路徑
ENV PYTHONPATH "${PYTHONPATH}:/app/yolov10"

# --- 安裝系統依賴 (包含 git-lfs) ---
RUN apt-get update && apt-get install -y \
    fonts-wqy-zenhei \
    git \
    git-lfs \
    python3.10 \
    python3-pip \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
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
    libqt5widgets5 \
    libqt5gui5 \
    libqt5dbus5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- 設定工作目錄 ---
WORKDIR /app

# --- 下載專案原始碼並處理依賴 ---
# 1. 複製主專案並拉取 LFS 檔案
RUN git clone https://github.com/yangtandev/face_system.git . && \
    git lfs pull && \
# 2. 複製 YOLOv10
    git clone https://github.com/THU-MIG/yolov10.git && \
# 3. 安裝所有 Python 套件
    pip3 install --no-cache-dir -r requirements.txt

# --- 設定啟動指令 ---
CMD ["python3", "main.py"]
