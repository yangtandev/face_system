# Face System

## 專案說明

本專案為人臉辨識系統。

## 環境準備

### 必要條件

-   Ubuntu 20.04 或更高版本
-   Git
-   Python 3.10+

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

本系統的所有預設設定都集中管理於 `setting/bulid_config.py` 檔案中的 `default_config` 字典。

**首次設定或需要修改預設值時，請直接編輯此檔案，並根據您的實際環境修改以下重要設定：**

*   **SSH 金鑰路徑**：
    `Server` -> `ssh_key_path`: 確保此路徑指向您正確的 SSH **私鑰**檔案。
    ```python
    # setting/bulid_config.py

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
    # setting/bulid_config.py

    "cameraIP": {
        "in_camera": "rtsp://your_camera_ip_here",      # <-- 修改成您的攝影機 IP
        "out_camera": "rtsp://your_other_camera_ip_here" # <-- 修改成您的攝影機 IP
    },
    ```

### 2. 生成 `config.json` 設定檔

完成 `setting/bulid_config.py` 的修改後，請執行以下指令來生成系統實際會讀取的 `config.json` 檔案：

```bash
python setting/bulid_config.py
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
`bash
    ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<server_ip>
    `
請將 `<username>` 和 `<server_ip>` 替換成您在 `setting/bulid_config.py` 中設定的伺服器使用者名稱與 IP。

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
    *   請務必將所有 `<您的專案絕對路徑>` 替換成您的 `face_system` **專案所在的絕對路徑** (例如：`/home/your_username/face_system`)。

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
    Environment=XAUTHORITY=/home/<您的使用者名稱>/.Xauthority
    # 新增 XDG_RUNTIME_DIR，解決部分桌面環境下的通訊問題
    # 注意：'1000' 是常見的預設使用者 ID，請根據您的系統作調整 (可用 `id -u <您的使用者名稱>` 查詢)
    Environment="XDG_RUNTIME_DIR=/run/user/1000"

    # 設定工作目錄
    WorkingDirectory=<您的專案絕對路徑>

    # 執行指令（使用絕對路徑）
    ExecStart=<您的專案絕對路徑>/venv/bin/python -u <您的專案絕對路徑>/face_main.py

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