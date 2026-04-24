# Face System

## 專案說明

本專案為人臉辨識系統。

## 環境準備

### 必要條件

- Ubuntu 20.04 或更高版本
- Git
- **全新的主機系統，極度推薦直接使用本文的「一鍵安裝腳本」**。
- (舊有提示) 確保您的系統可以連網並具備基本的編譯環境。大型模型將透過 Git LFS 下載，程式串流解碼則依賴 FFmpeg。

## 一鍵全自動部署 (全新推薦)

本專案提供 `install.sh` 自動化部署腳本。此腳本實作了 **100% Zero-Touch 的無人值守安裝**，解決了手動安裝時版本衝突或路徑設錯的困擾。它會自動完成：系統基礎套件安裝、Python 虛擬環境配置、組態檔生成、SSH 免密碼遠端連線佈署（含金鑰產生與指紋略過）、Systemd 常駐服務註冊，以及報表系統排程。

1.  **克隆專案**：

    ```bash
    git clone https://github.com/yangtandev/facial_recognition.git
    cd facial_recognition
    ```

2.  **執行一鍵安裝腳本**：
    _(請注意：請勿直接使用 sudo 執行，只需一般使用者身分執行即可。腳本內部會在需要時，透過腳本開頭的授權自動針對特定指令提權，以保障您的專案檔案與虛擬環境權限歸屬正確)_

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

3.  **依提示輸入參數**：
    腳本開始時會請您輸入基礎資訊（伺服器 IP、RTSP 網址、SSH 登入密碼，以及**進出螢幕解析度**）。
    
    **新增功能：螢幕配置檢測**
    - 安裝腳本會自動偵測您的**攝影機配置**（單鏡頭或雙鏡頭）。
    - 如果 IN Camera 與 OUT Camera 相同，系統會自動進入**單螢幕模式**，加快啟動並由 PyQt5 自行處理視窗佈局。
    - 如果配置不同，系統會進入**雙螢幕模式**，使用 `wmctrl` 工具精準定位視窗位置。
    
    填寫完畢後，腳本將完全接管後續流程，直到服務成功啟動與註冊完畢為止。

---

## 屏幕配置模式 (單/雙螢幕支援)

本系統支援**靈活的單螢幕與雙螢幕部署**，自動適應您的硬體環境：

### 自動檢測機制

- **雙攝影機配置**（IN Camera ≠ OUT Camera）：
  - 系統進入**雙螢幕模式**
  - 服務啟動時會等待 15 秒讓硬體初始化
  - 使用 `wmctrl` 自動定位視窗：進入視窗在左側，離開視窗在右側
  - 適合需要同時監控進出口的場景

- **單攝影機配置**（IN Camera = OUT Camera）：
  - 系統進入**單螢幕模式**
  - 服務啟動時只需等待 5 秒
  - 由 PyQt5 自行處理視窗佈局，支援上下或左右自適應排列
  - 適合臨時測試或單一區域監控

### 配置方式

1. **安裝時設定**：執行 `./install.sh` 時填入攝影機 RTSP 網址，系統會自動判斷
2. **安裝後修改**：透過 GUI 設定工具或 Web 介面隨時調整攝影機與螢幕配置，系統自動同步至服務

---

## 系統使用與管理

（由腳本全自動部署完成後，您的系統已具備 Web 與 GUI 兩種設定方式，並已設為開機自動啟動。）

### 1. 圖形化設定工具 (GUI Config Tool)

本系統提供了一個方便的圖形化介面，讓您可以無需編輯 JSON 檔案即可調整各項參數。

**啟動方式：**

1.  **從主畫面啟動**：在人臉辨識主視窗的左上角，點擊 **齒輪圖示**。輸入管理員密碼（預設：`admin`）即可開啟。
2.  **獨立啟動**：
    ```bash
    cd facial_recognition
    source venv/bin/activate
    python ui/setting_tool.py
    ```

**功能特色：**

- **一般設定**
  - 修改攝影機 RTSP 網址
  - **螢幕配置**：設定進入/離開螢幕的解析度（格式：`寬x高`，例如 `1080x1920`）
  - 切換介面主題 (Dark/Light)
  - 開機自動啟動選項

- **排程設定**：針對單鏡頭場景，設定多個自動切換「入口/出口」的時段。

- **熱更新 (Soft Reload)**
  - 修改設定後，系統會自動重新載入而不中斷服務 (PID 不變)
  - **自動同步至 Service**：修改螢幕配置後會自動更新 systemd service 檔案並重新載入

### 2. 網頁版設定後台 (Web Config Interface)

除了本機的圖形化工具外，本系統也內建了輕量級的網頁設定介面，方便遠端管理。

**使用方式：**

1.  確保您的裝置與主機在同一區域網路 (LAN) 下。
2.  打開瀏覽器，輸入：`http://<主機IP>:5000` (例如 `http://192.168.1.100:5000`)。
3.  登入密碼預設為：`admin`。

_(Web 介面對 Systemd 服務的強制重啟權限，已經由 `install.sh` 自動設定完備，無需手動微調)_

**功能特色：**

- **跨平台支援**：手機、平板、筆電皆可使用。
- **設定管理**：
  - 遠端修改攝影機、伺服器、辨識等各項參數
  - **即時調整螢幕解析度**：無需重啟即可修改進出螢幕配置
- **即時日誌檢視**：可直接查看應用程式日誌 (App Log) 與系統日誌 (System Log)。
- **遠端重啟**：支援軟重啟 (Soft Reload) 與強制重啟 (Hard Restart)。
- **自動 Service 同步**：修改配置後自動更新 systemd 服務設定

---

## 系統日誌與每日報告

應用程式的日誌會統一輸出到 `log` 資料夾中，主要日誌檔案為 `log/facial_recognition.log`。

### 每日辨識成效報告

系統內建 `analytics_reporter.py` 分析工具，且**已預設由自動腳本寫入主機 Crontab 中**（每小時自動執行一次）。您可以在 `reports/` 目錄中找到相關報表：

1.  **`report-YYYY-MM-DD.txt`**: 每日詳細報告。每次執行都會將最新的統計結果追加到當天的日報檔案中。
2.  **`data-YYYY-MM-DD.json`**: 每日原始數據。供圖表與趨勢分析串接使用。
3.  **`summary_7_days.txt`**: 滾動式七日總結報告。每天更新，分析過去七天的辨識成效趨勢。

（腳本會自動清理超過七天的舊日報與數據檔案。）

---

---

## <details><summary><b>舊版手動安裝與進階設定 (除錯/開發用指南)</b></summary>

若您因為特殊原因無法使用 `install.sh` 一鍵安裝，或者正在進行開發與除錯，可參考以下原始的手動架構設定步驟：

### 1. 安裝系統相依套件與拉取 Git LFS

1.  **安裝 Git LFS 與 FFmpeg 工具**：

    ```bash
    sudo apt-get update && sudo apt-get install -y git-lfs ffmpeg mosquitto mosquitto-clients
    git lfs install
    git lfs pull
    ```

2.  **建立 Python 虛擬環境包**：

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **開啟 MQTT 服務**：
    ```bash
    sudo systemctl enable mosquitto && sudo systemctl start mosquitto
    ```

### 2. 生成 `config.json` 設定檔

編輯 `setting/build_config.py` 中的 `default_config` 字典，修改您的 `Server IP`, `User`, `SSH Path` 以及 `Camera RTSP`。修改完畢後，執行以下指令生成系統實際讀取的 `config.json` 檔案：

```bash
python setting/build_config.py
```

### 3. SSH 金鑰設定與佈署

系統使用 SSH 金鑰來安全地與遠端伺服器 (EC2) 連線。

1. **產生新的 SSH 金鑰**：

    ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```

2. **複製公鑰至遠端伺服器**：
    ```bash
    ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<server_ip>
    ```

### 4. 手動註冊為 Systemd 用戶級服務 (開機自啟動)

1.  **建立服務目錄與文件**:

    ```bash
    mkdir -p ~/.config/systemd/user
    nano ~/.config/systemd/user/facial_recognition.service
    ```

2.  **貼入以下內容**（請替換 `<PROJECT_DIR>`、`<USER_ID>` 與 `<HOME>` 為實際路徑）：

    > `<PROJECT_DIR>` 範例：`/home/ubuntu/facial_recognition`  
    > `<USER_ID>` 範例：`1000`（執行 `id -u` 取得）  
    > `<HOME>` 範例：`/home/ubuntu`

    > **自動螢幕偵測**：Systemd 僅負責啟動 Python 程序。單/雙螢幕的偵測、解析度讀取與視窗定位，已全部由 Python 程式在啟動時（或斷電重啟後）自動偵測處理，無需在 service 檔中額外設定。

    ```ini
    [Unit]
    Description=Face Recognition System User Service
    After=network.target graphical-session.target

    [Service]
    Type=simple
    Environment=DISPLAY=:0
    Environment=XDG_RUNTIME_DIR=/run/user/<USER_ID>
    Environment=XAUTHORITY=<HOME>/.Xauthority
    Environment=PYTHONUNBUFFERED=1
    WorkingDirectory=<PROJECT_DIR>
    ExecStart=<PROJECT_DIR>/venv/bin/python -u <PROJECT_DIR>/main.py
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=graphical-session.target
    ```

    > **提示**：雙螢幕模式需安裝 `wmctrl`：`sudo apt-get install -y wmctrl`

3.  **重新載入並啟動服務**:

    ```bash
    systemctl --user daemon-reload
    systemctl --user enable facial_recognition.service
    systemctl --user start facial_recognition.service
    ```

### 5. Web 介面重啟權限

Web 介面現在使用用戶級服務，無需額外 sudo 權限設定。

### 6. 配置監聽與自動同步 (v2026+)

系統內建 **ConfigWatcher** 機制，自動監聽 `config.json` 的變化：

- 任何透過 GUI 或 Web 介面進行的配置修改會自動被偵測
- 螢幕配置（進入/離開解析度）變更後會自動重新生成 systemd service 檔案
- 無需手動執行 `systemctl --user daemon-reload` 與 `restart`
- 系統會在後台執行緒定期檢查，確保配置同步
  
如欲查看監聽日誌，可執行：
```bash
journalctl --user -u facial_recognition.service -f
```

</details>
