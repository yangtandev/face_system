# 專案：Face System 部署與維護手冊

**版本：2.0 (Docker-based)**
**最後更新：2025-08-12**

---

## 1. 專案簡介

本專案為一套基於 Python 的人臉與安全服裝辨識系統，用於現場主機的本地化部署。為了統一開發與生產環境、簡化在客戶端 Ubuntu 主機上的部署流程，現已全面遷移至 Docker 容器化部署。

---

## 2. 部署流程

### 2.1. 環境需求

-   **作業系統**: Ubuntu 22.04 Desktop (或具備圖形介面的 Linux 發行版)
-   **軟體**: Docker Engine, Docker Compose, Git

### 2.2. 首次環境設定

1.  **安裝 Docker 與 Docker Compose** (參考官方文件)。
2.  **拉取專案程式碼**: `git clone <your-git-repository-url> && cd face_system`

### 2.3. 參數設定 (`config.json`)

這是部署前**唯一**需要手動修改的檔案。它控制了所有與環境相關的變數。

**完整參數說明：**

```json
{
	"ip_set": {
		// 設定機器IP(非強制)
		"ip_address": "",
		"ip_gateway": "",
		"ip_mask": ""
	},
	"cameraIP": {
		// 設定攝影機IP
		"in_camera": "rtsp://@192.168.2.60",
		"out_camera": "rtsp://admin:!QAZ87518499@192.168.31.59:554"
	},
	"Server": {
		// EC2伺服器參數
		"ip": "54.92.8.82",
		"username": "ubuntu",
		"password": "875184991qaz2wsx",
		"face_data_dir": "/home/ubuntu/pvms-api/media", // 人臉檔案位置
		"API_url": "https://demosite.api.ginibio.com/api/v1",
		"location_ID": 1
	},
	"say": {
		// 語音內容
		"in": "簽到",
		"out": "簽離",
		"clothes": "請正確著裝"
	},
	"inCamera": {
		// 入口鏡頭
		"close": false // 範圍放大
	},
	"outCamera": {
		// 出口鏡頭
		"close": false // 範圍放大
	},
	"door": "0", // http://{門禁控制器IP}:1880/open_door
	"Clothes_detection": false, // 是否開啟服辨
	"Clothes_show": false,
	"min_face": 130, // 最小人臉大小
	"test_mod": false, // 工程除錯模式
	"auto_open": false, // 開啟程式後自動啟動
	"full_screen": false // 全螢幕
}
```
