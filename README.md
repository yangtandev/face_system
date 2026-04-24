# Facial Recognition System

## Project Description

This project is a comprehensive facial recognition system designed for edge computing environments.

## Environment Preparation

### Prerequisites

- Ubuntu 20.04 or higher
- Git
- **Fresh OS installation is highly recommended to use the "One-Click Install Script" directly.**

## One-Click Automatic Deployment (Recommended)

This project provides an `install.sh` script for automated deployment. This script implements **100% Zero-Touch unattended installation**, avoiding issues with version conflicts or incorrect paths. It automatically completes: system package installation, Python virtual environment configuration, configuration file generation, SSH key-based remote connection deployment, Systemd service registration, and automated reporting.

1.  **Clone the project**:

    ```bash
    git clone https://github.com/yangtandev/facial_recognition.git
    cd facial_recognition
    ```

2.  **Run the install script**:
    _(Note: Do not use `sudo` directly; run as a regular user. The script will request elevation only when necessary.)_

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

3.  **Input parameters when prompted**:
    The script will ask for basic information (Server IP, RTSP URL, SSH password, and **Screen Resolutions**).
    
    **New Feature: Screen Configuration Detection**
    - The script automatically detects your **camera configuration** (Single or Dual camera).
    - If the IN and OUT cameras are the same, the system enters **Single Screen Mode**, where PyQt5 handles the layout automatically.
    - If they are different, the system enters **Dual Screen Mode**, using the `wmctrl` tool for precise window positioning.
    
    Once information is provided, the script will handle the rest until the service is successfully started and registered.

---

## Screen Configuration Mode (Single/Dual Screen Support)

The system supports **flexible single and dual screen deployment**, automatically adapting to your hardware:

### Automatic Detection Mechanism

- **Dual Camera Configuration** (IN Camera ≠ OUT Camera):
  - System enters **Dual Screen Mode**.
  - Service waits 15 seconds for hardware initialization at startup.
  - Uses `wmctrl` to position windows: Entry on the left, Exit on the right.
  - Ideal for scenarios requiring simultaneous monitoring of entry and exit.

- **Single Camera Configuration** (IN Camera = OUT Camera):
  - System enters **Single Screen Mode**.
  - Service waits only 5 seconds at startup.
  - PyQt5 handles window layout with adaptive vertical or horizontal alignment.
  - Ideal for testing or single-area monitoring.

### Configuration Methods

1. **At Installation**: Provided via `./install.sh` prompts.
2. **Post-Installation**: Adjusted via the GUI Setting Tool or Web Interface; the system synchronizes changes to the service automatically.

---

## System Usage and Management

(After automated deployment, the system supports both Web and GUI configuration methods and is set to start automatically on boot.)

### 1. GUI Config Tool

The system provides a convenient graphical interface to adjust parameters without editing JSON files.

**How to start:**

1.  **From the Main Screen**: Click the **Gear Icon** in the top-left corner of the recognition window. Enter the administrator password (default: `admin`) to open.
2.  **Standalone Start**:
    ```bash
    cd facial_recognition
    source venv/bin/activate
    python ui/setting_tool.py
    ```

**Key Features:**

- **General Settings**
  - Modify Camera RTSP URLs.
  - **Screen Configuration**: Set entry/exit screen resolutions (Format: `WidthxHeight`, e.g., `1080x1920`).
  - Toggle Interface Themes (Dark/Light).
  - Enable/Disable Auto-start on boot.

- **Scheduling**: For single-camera scenarios, set time slots to automatically switch between "Entry" and "Exit" modes.

- **Soft Reload**
  - Settings take effect immediately without stopping the service (PID remains the same).
  - **Auto-Sync to Service**: Changing screen configurations automatically updates the systemd service file and reloads the daemon.

### 2. Web Config Interface

A lightweight web interface is built-in for remote management.

**How to use:**

1.  Ensure your device is on the same Local Area Network (LAN) as the host.
2.  Open a browser and enter: `http://<HOST_IP>:5000` (e.g., `http://192.168.1.100:5000`).
3.  The default login password is: `admin`.

_(The Web interface's permission to restart Systemd services is pre-configured by `install.sh`)_

**Key Features:**

- **Cross-Platform Support**: Accessible via mobile, tablet, or laptop.
- **Configuration Management**:
  - Remotely modify camera, server, and recognition parameters.
  - **Instant Resolution Adjustment**: Modify screen configurations without a restart.
- **Real-time Log Viewer**: View App Logs and System Logs directly.
- **Remote Restart**: Supports Soft Reload and Hard Restart.
- **Automatic Service Sync**: Updates systemd service settings automatically upon configuration changes.

---

## System Logs and Daily Reports

Application logs are stored in the `log` folder, primarily in `log/facial_recognition.log`.

### Daily Performance Reports

The built-in `analytics_reporter.py` tool is scheduled via Crontab (running once every hour) by the install script. Reports are located in the `reports/` directory:

1.  **`report-YYYY-MM-DD.txt`**: Detailed daily report. Statistical results are appended to the daily file during each run.
2.  **`data-YYYY-MM-DD.json`**: Daily raw data for charting and trend analysis.
3.  **`summary_7_days.txt`**: Rolling 7-day summary report updated daily to analyze recognition performance trends.

(The script automatically cleans up daily reports and data files older than seven days.)

---

## <details><summary><b>Manual Installation & Advanced Configuration (Developer Guide)</b></summary>

If you cannot use the `install.sh` script or are performing development/debugging, refer to these manual steps:

### 1. Install Dependencies and Git LFS

1.  **Install Git LFS and FFmpeg**:
    ```bash
    sudo apt-get update && sudo apt-get install -y git-lfs ffmpeg mosquitto mosquitto-clients
    git lfs install
    git lfs pull
    ```

2.  **Setup Python Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Enable MQTT Service**:
    ```bash
    sudo systemctl enable mosquitto && sudo systemctl start mosquitto
    ```

### 2. Generate `config.json`

Edit the `default_config` dictionary in `setting/build_config.py` with your `Server IP`, `User`, `SSH Path`, and `Camera RTSP`. Then run:

```bash
python setting/build_config.py
```

### 3. SSH Key Configuration

The system uses SSH keys for secure communication with the remote server.

1. **Generate a new SSH key**:
    ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```

2. **Copy the public key to the server**:
    ```bash
    ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<server_ip>
    ```

### 4. Manual Systemd Registration (Auto-start)

1.  **Create Service Directory and File**:
    ```bash
    mkdir -p ~/.config/systemd/user
    nano ~/.config/systemd/user/facial_recognition.service
    ```

2.  **Paste Content** (Replace `<PROJECT_DIR>`, `<USER_ID>`, and `<HOME>` with actual paths):

    > Example `<PROJECT_DIR>`: `/home/ubuntu/facial_recognition`
    > Example `<USER_ID>`: `1000` (Get via `id -u`)
    > Example `<HOME>`: `/home/ubuntu`

    > **Auto Screen Detection**: Systemd only starts the Python process. Screen detection, resolution reading, and window positioning are handled automatically by the Python application at startup.

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

    > **Note**: Dual-screen mode requires `wmctrl`: `sudo apt-get install -y wmctrl`

3.  **Reload and Start Service**:
    ```bash
    systemctl --user daemon-reload
    systemctl --user enable facial_recognition.service
    systemctl --user start facial_recognition.service
    ```

### 5. Web Interface Restart Permission

Web interfaces now run as user-level services and do not require additional sudo configuration for restarts.

### 6. Config Monitoring and Auto-Sync (v2026+)

The system includes a **ConfigWatcher** mechanism that monitors changes to `config.json`:

- Any configuration changes via GUI or Web are automatically detected.
- Systemd service files are regenerated automatically upon screen resolution changes.
- No manual `daemon-reload` or `restart` is required.
- A background thread ensures configuration synchronization.

To view monitoring logs:
```bash
journalctl --user -u facial_recognition.service -f
```

</details>
