# Face System

## 專案說明

本專案為人臉辨識系統，使用 Docker 進行容器化部署。

## 快速啟動

### 必要條件

-   Ubuntu 20.04 或更高版本
-   Git
-   Docker
-   Docker Compose

### 一鍵安裝與啟動

在您的 Ubuntu 主機上，執行以下命令即可自動完成所有設定並啟動專案：

```bash
bash <(curl -s https://raw.githubusercontent.com/yangtandev/face_system/main/setup.sh)
```

或者，您也可以手動執行以下步驟：

1.  **克隆專案**：

    ```bash
    git clone https://github.com/yangtandev/face_system.git
    cd face_system
    ```

2.  **執行安裝腳本**：
    ```bash
    bash setup.sh
    ```

### 查看日誌

您可以使用以下命令來查看應用程式的日誌：

```bash
docker compose logs -f
```

### 停止應用程式

若要停止應用程式，請執行：

```bash
docker compose down
```
