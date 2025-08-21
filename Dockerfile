# --- 基礎映像檔 ---
FROM ubuntu:22.04

# --- 設定環境變數 ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# 解決 RTSP (HEVC) 影像流可能出現的 "VPS 0 does not exist" 錯誤
# 強制 OpenCV 的 FFmpeg 後端使用 TCP 協定來接收 RTSP 影像流，提高連線穩定性
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
# 將 yolov10 目錄加入 Python 的模組搜尋路徑
ENV PYTHONPATH "${PYTHONPATH}:/app/yolov10"

# --- 安裝系統依賴 (包含 git-lfs, rsync, ping) ---
RUN apt-get update && apt-get install -y \
    fonts-wqy-zenhei \
    git \
    git-lfs \
    rsync \
    iputils-ping \
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

# --- 複製本地專案檔案到容器中 ---
COPY . .

# --- 處理依賴 ---
# 1. 拉取 LFS 檔案
# 2. 複製 YOLOv10
# 3. 安裝所有 Python 套件
RUN git lfs pull && \
    git clone https://github.com/THU-MIG/yolov10.git && \
    pip3 install --no-cache-dir -r requirements.txt

# --- 設定啟動指令 ---
# 注意：此 CMD 為預設指令，實際上會被 docker-compose.yml 中的 command 覆寫
CMD ["python3", "face_main.py"]
