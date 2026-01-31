# Face System

## 專案說明

本專案為人臉辨識系統。

## 環境準備

### 必要條件

-   Ubuntu 20.04 或更高版本
-   Git
-   **Git LFS (用於下載大型模型權重檔)**
-   **FFmpeg (用於影像解碼與處理)**
-   Python 3.10+

#### 安裝 Git LFS

由於本專案的模型權重檔案 (`.pt`) 較大，使用 Git LFS 進行管理。在執行 `git clone` 之前或之後，請確保您的系統已安裝 Git LFS：

```bash
# 安裝 Git LFS 套件
sudo apt-get update && sudo apt-get install -y git-lfs

# 初始化 Git LFS (每個系統使用者只需執行一次)
git lfs install

# 如果您已經克隆了專案，請執行以下指令下載實際的模型檔案
git lfs pull
```

#### 安裝 FFmpeg

系統依賴 FFmpeg 進行攝影機串流的解碼。請執行以下指令安裝：

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## 安裝與設定

1.  **克隆專案**：

    ```bash
    git clone https://github.com/yangtandev/face_system.git
    cd face_system
    ```

2.  **安裝依賴套件**：
    建議在虛擬環境中安裝。
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## 系統設定

在執行應用程式之前，您必須完成以下設定。

### 1. 修改預設設定

本系統的所有預設設定都集中管理於 `setting/build_config.py` 檔案中的 `default_config` 字典。

**首次設定或需要修改預設值時，請直接編輯此檔案，並根據您的實際環境修改以下重要設定：**

*   **SSH 金鑰路徑**：
    `Server` -> `ssh_key_path`: 確保此路徑指向您正確的 SSH **私鑰**檔案。
    ```python
    # setting/build_config.py

    "Server": {
        "ip": "43.213.128.240",
        "username": "ubuntu",
        "ssh_key_path": "/home/your_username/.ssh/id_rsa",  # <-- 修改成您的路徑
        # ...
    },
    ```

*   **攝影機 IP**：
    `cameraIP`: 修改 `in_camera` 和 `out_camera` 的值，填入您實際的 RTSP 串流 URL。
    ```python
    # setting/build_config.py

    "cameraIP": {
        "in_camera": "rtsp://your_camera_ip_here",      # <-- 修改成您的攝影機 IP
        "out_camera": "rtsp://your_other_camera_ip_here" # <-- 修改成您的攝影機 IP
    },
    ```

### 2. 生成 `config.json` 設定檔

完成 `setting/build_config.py` 的修改後，請執行以下指令來生成系統實際會讀取的 `config.json` 檔案：

```bash
python setting/build_config.py
```

`config.json` 檔案會被 `.gitignore` 忽略，不應手動修改或提交至版本控制。

### 3. 設定 SSH 金鑰 (SSH Key)

系統使用 SSH 金鑰來安全地與遠端伺服器連線，取代了不安全的明碼密碼。

#### 如何產生與使用 SSH 金鑰？

i. **檢查現有金鑰**：
首先，檢查您是否已經有 SSH 金鑰。

```bash
ls -al ~/.ssh/id_rsa*
```

如果看到 `id_rsa` 和 `id_rsa.pub` 檔案，表示您已有金鑰，可以跳至步驟 iii。

ii. **產生新的 SSH 金鑰**：
如果沒有金鑰，請執行以下命令產生一對新的 4096 位元 RSA 金鑰。過程中可以直接按 Enter 使用預設設定。

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

這會在 `~/.ssh/` 目錄下產生 `id_rsa` (私鑰) 和 `id_rsa.pub` (公鑰) 兩個檔案。

iii. **複製公鑰至遠端伺服器**：
您需要將您的**公鑰** (`id_rsa.pub`) 的內容附加到遠端伺服器的 `~/.ssh/authorized_keys` 檔案中。最簡單的方法是使用 `ssh-copy-id` 命令：

```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<server_ip>
```

請將 `<username>` 和 `<server_ip>` 替換成您在 `setting/build_config.py` 中設定的伺服器使用者名稱與 IP。

### 4. 設定與安裝 MQTT Broker (Mosquitto)

本系統利用 MQTT 協定來接收來自伺服器的即時更新通知（如人員變更）。因此，您需要在本地機器上安裝並啟動 Mosquitto MQTT Broker。

執行以下指令一次性完成安裝、啟用與啟動：

```bash
sudo apt-get update && sudo apt-get install -y mosquitto mosquitto-clients && sudo systemctl enable mosquitto && sudo systemctl start mosquitto && sudo systemctl status mosquitto --no-pager
```

**驗證安裝：**
安裝完成後，您可以使用 `systemctl status mosquitto` 檢查服務狀態，應顯示為 `active (running)`。

### 5. 圖形化設定工具 (GUI Config Tool)

本系統提供了一個方便的圖形化介面，讓您可以無需編輯 JSON 檔案即可調整各項參數。

**啟動方式：**
1.  **從主畫面啟動**：在人臉辨識主視窗的左上角，點擊 **齒輪圖示 (⚙)**。輸入管理員密碼（預設：`admin`）即可開啟。
2.  **獨立啟動**：
    ```bash
    python setting_tool.py
    ```

**功能特色：**
*   **一般設定**：修改攝影機 RTSP、切換介面主題 (Dark/Light)。
*   **排程設定**：針對單鏡頭場景，設定多個自動切換「入口/出口」的時段。
*   **熱更新 (Soft Reload)**：修改設定後，系統會自動重新載入而不中斷服務 (PID 不變)。

## 執行應用程式

完成所有設定後，您可以使用以下命令來啟動主程式：

```bash
python face_main.py
```

## 查看日誌

應用程式的日誌會輸出到 `log` 資料夾中，主要的日誌檔案是 `log/face_system.log`。

### 設定為系統服務 (開機自啟動) (可選)

如果您希望此臉部辨識應用程式在系統開機並登入圖形介面後自動啟動，可以將其設定為一個 `systemd` 服務。

1.  **建立服務文件**:
    使用 `nano` 或您偏好的編輯器，建立一個新的服務設定檔。

    ```bash
    sudo nano /etc/systemd/system/face_system.service
    ```

2.  **貼入以下內容**:
    將下面的設定內容完整複製並貼到編輯器中。

    **重要提示：**
    *   請務必將所有 `<您的使用者名稱>` 替換成您自己的 Linux **使用者名稱** (例如：`ubuntu`, `your_username`)。
    *   請務必將所有 `/home/<您的使用者名稱>/face_system` 替換成您的 `face_system` **專案所在的絕對路徑** (例如：`/home/your_username/face_system`)。

    ```ini
    [Unit]
    Description=Face Recognition System GUI Application
    # 我們需要等待圖形介面登入管理器啟動
    After=graphical.target

    [Service]
    # *** 關鍵改動 1: 指定正確的使用者 ***
    # 這解決了 status=203/EXEC 的權限問題
    User=<您的使用者名稱>

    # *** 關鍵改動 2: 指定環境變數 ***
    # 這讓服務知道要在哪個螢幕、哪個使用者會話中顯示 GUI
    Environment=DISPLAY=:0
    # 新增 XDG_RUNTIME_DIR，解決部分桌面環境下的通訊問題 (特別是 PulseAudio 音訊輸出)
    # 注意：'1000' 是常見的預設使用者 ID，請根據您的系統作調整 (可用 `id -u <您的使用者名稱>` 查詢)
    Environment="XDG_RUNTIME_DIR=/run/user/1000"

    # 設定工作目錄
    WorkingDirectory=/home/<您的使用者名稱>/face_system

    # 執行指令（使用絕對路徑）
    ExecStart=/home/<您的使用者名稱>/face_system/venv/bin/python -u /home/<您的使用者名稱>/face_system/face_main.py

    # 自動重啟設定
    Restart=always
    RestartSec=10

    [Install]
    # *** 關鍵改動 3: 綁定到正確的系統級目標 ***
    # graphical.target 是系統級的圖形介面目標
    WantedBy=graphical.target
    ```

3.  **重新載入、啟用並啟動服務**:
    儲存並關閉檔案後，執行以下指令來讓 `systemd` 讀取新的設定，並設定為開機自啟動。

    ```bash
    # 重新載入 systemd 設定
    sudo systemctl daemon-reload

    # 啟用服務 (設定為開機自啟動)
    sudo systemctl enable face_system.service

    # 立刻啟動服務
    sudo systemctl start face_system.service
    ```

4.  **檢查服務狀態**:
    您可以隨時使用以下指令來檢查服務的運行狀態。

    ```bash
    sudo systemctl status face_system.service
    ```

5.  **查看服務日誌**:
    若要即時查看由 `systemd` 執行的應用程式所產生的日誌，請使用 `journalctl`。

    ```bash
    journalctl -u face_system.service -f
    ```

## 疑難排解 (Troubleshooting)

### 錯誤：`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`

如果在執行程式時遇到關於 `xcb` 找不到或無法載入的錯誤，這通常表示系統缺少 Qt 圖形介面所依賴的函式庫。

**解決方案**：
在基於 Debian/Ubuntu 的系統上，執行以下指令來安裝必要的依賴套件：

```bash
sudo apt-get update && sudo apt-get install -y libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0
```

## 每日辨識成效報告設定 (Daily Recognition Performance Report Setup)

為了持續監控系統的辨識效能與趨勢，我們新增了一個自動化的報告工具 (`analytics_reporter.py`)。它會定期執行，分析日誌，並產生詳細的報告。

### 報告產出

腳本執行後，會將報告存儲在專案根目錄下的 `reports` 資料夾中：
1.  **`report-YYYY-MM-DD.txt`**: 每日詳細報告。每次執行都會將最新的統計結果（帶有精確時間戳記）追加到當天的日報檔案中。
2.  **`data-YYYY-MM-DD.json`**: 每日原始數據。以 JSON 格式保存當天的統計數據，每次執行會覆寫最新的數據，供趨勢分析使用。
3.  **`summary_7_days.txt`**: 滾動式七日總結報告。這份報告每天都會更新，分析並總結過去七天的辨識成效趨勢。

腳本會自動清理超過七天的舊日報與數據檔案。

### 排程設定 (推薦使用 Cron)

建議使用 Linux 系統的 `cron` 排程服務，讓 `analytics_reporter.py` 腳本每小時自動執行一次。

1.  **打開 Crontab 編輯器**：
    在終端機中輸入 `crontab -e`，然後按 Enter。

2.  **新增排程條目**：
    在打開的文字編輯器中，將以下這一整行指令複製並貼上到檔案的最後。

    **重要提示：**
    *   請務必將 `/home/<您的使用者名稱>/face_system` 替換成您的 `face_system` **專案所在的絕對路徑** (例如：`/home/your_username/face_system`)。您可以使用 `pwd` 命令在專案根目錄下查詢。

    ```bash
    0 * * * * /home/<您的使用者名稱>/face_system/venv/bin/python /home/<您的使用者名稱>/face_system/analytics_reporter.py >> /home/<您的使用者名稱>/face_system/log/reporter.log 2>&1
    ```
    替換後的範例：
    ```bash
    0 * * * * /home/your_username/face_system/venv/bin/python /home/your_username/face_system/analytics_reporter.py >> /home/your_username/face_system/log/reporter.log 2>&1
    ```

3.  **儲存並關閉**：
    儲存並關閉編輯器。Cron 服務會自動讀取新的設定。

### 查看報告

*   **每日詳細報告**：請查看 `reports/report-YYYY-MM-DD.txt`。
*   **七日總結報告**：請查看 `reports/summary_7_days.txt`。
*   **報表腳本日誌**：請查看 `log/reporter.log`
