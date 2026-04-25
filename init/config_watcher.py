import os
import subprocess
from pathlib import Path


class ConfigWatcher:
    """生成與管理 systemd user-level service 文件。

    Systemd 只負責啟動 Python，螢幕偵測與視窗定位由 Python/Qt 在運行時處理。
    因此 service 模板是固定的，不依賴任何 config.json 欄位。
    """

    def __init__(self, config_path, service_name="facial_recognition.service"):
        self.config_path = os.path.abspath(config_path)
        self.service_name = service_name

    def _build_service_content(self):
        """生成固定的 systemd service 內容。

        不包含任何 bash 包裝、sleep、或 xrandr 邏輯。
        螢幕偵測與視窗定位由 Python/Qt 在程式啟動時自動處理。
        """
        project_dir = os.path.dirname(self.config_path)
        user_id = os.getuid()
        home_dir = Path.home()
        python_exec = f"{project_dir}/venv/bin/python"

        return (
            f"[Unit]\n"
            f"Description=Face Recognition System User Service\n"
            f"After=network.target graphical-session.target\n"
            f"\n"
            f"[Service]\n"
            f"Type=simple\n"
            f"Environment=DISPLAY=:0\n"
            f"Environment=XDG_RUNTIME_DIR=/run/user/{user_id}\n"
            f"Environment=XAUTHORITY={home_dir}/.Xauthority\n"
            f"Environment=PYTHONUNBUFFERED=1\n"
            f"WorkingDirectory={project_dir}\n"
            f"ExecStart={python_exec} -u {project_dir}/main.py\n"
            f"Restart=always\n"
            f"RestartSec=10\n"
            f"\n"
            f"[Install]\n"
            f"WantedBy=graphical-session.target\n"
        )

    def update_service(self):
        """更新 systemd service 文件並執行 daemon-reload。

        若 service 文件內容無變化則跳過。
        """
        try:
            new_content = self._build_service_content()

            service_dir = Path.home() / '.config' / 'systemd' / 'user'
            service_dir.mkdir(parents=True, exist_ok=True)
            service_file = service_dir / self.service_name

            # 比對現有內容，若無變化則跳過
            if service_file.exists():
                with open(service_file, 'r', encoding='utf-8') as f:
                    if f.read() == new_content:
                        print("Service 文件無變化，跳過更新")
                        return True

            # 寫入新的 service 文件
            with open(service_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            print("Service 文件已更新並 daemon-reload")
            return True

        except Exception as e:
            print(f"更新 service 失敗: {e}")
            return False
