import sys
import os

# [Fix] Add project root to sys.path to allow imports from 'ui' package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QLockFile, QDir
import ui.config_window # Import module to access IS_RESTARTING
from ui.config_window import ConfigWindow
from ui import styles
import threading, time, argparse, json, os

def parent_watchdog(initial_ppid):
    """Monitor parent process. Exit if parent dies, unless restarting."""
    while True:
        current_ppid = os.getppid()
        if current_ppid != initial_ppid:
            # Parent changed (likely died)
            if not ui.config_window.IS_RESTARTING:
                print(f"Parent process {initial_ppid} died. Closing setting tool.")
                QApplication.quit()
            break
        time.sleep(1)

def main():
    # 設定環境變數以支援高 DPI (選用)
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    # [2026-01-30 Fix] Linux/WSL Chinese Input Method Support
    if sys.platform.startswith("linux"):
        # Try to use fcitx by default if not set
        if "QT_IM_MODULE" not in os.environ:
            os.environ["QT_IM_MODULE"] = "fcitx"
        if "GTK_IM_MODULE" not in os.environ:
            os.environ["GTK_IM_MODULE"] = "fcitx"
        if "XMODIFIERS" not in os.environ:
            os.environ["XMODIFIERS"] = "@im=fcitx"
    
    app = QApplication(sys.argv)
    
    # [2026-01-30 Feature] Single Instance Protection
    # Use QLockFile to prevent multiple setting windows
    lock_file = QLockFile(QDir.temp().filePath("face_system_setting.lock"))
    if not lock_file.tryLock(100): # Try to lock for 100ms
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Warning)
        error_box.setWindowTitle("警告")
        error_box.setText("設定工具已在運行中！")
        error_box.setInformativeText("請檢查是否已有開啟的設定視窗。")
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()
        return

    # [2026-01-30 Feature] Apply Theme
    theme = "dark"
    try:
        config_path = os.path.join(os.path.dirname(__file__), "../config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            theme = cfg.get("theme", "dark")
    except Exception: pass
    
    app.setStyleSheet(styles.get_stylesheet(theme))
    
    window = ConfigWindow()
    
    # [2026-01-30 Fix] Center relative to parent if args provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_x", type=int, default=None)
    parser.add_argument("--parent_y", type=int, default=None)
    parser.add_argument("--parent_w", type=int, default=None)
    parser.add_argument("--parent_h", type=int, default=None)
    args, unknown = parser.parse_known_args()
    
    if args.parent_x is not None:
        # Calculate center
        # Child size (assuming window.resize() was called in __init__)
        # Note: Before show(), geometry might include frame or not depending on OS
        # We use a safe approximation or force resize
        cw = window.width()
        ch = window.height()
        
        target_x = args.parent_x + (args.parent_w - cw) // 2
        target_y = args.parent_y + (args.parent_h - ch) // 2
        
        window.move(target_x, target_y)
    
    window.show()
    
    # [2026-01-30 Feature] Start Parent Watchdog
    ppid = os.getppid()
    t = threading.Thread(target=parent_watchdog, args=(ppid,), daemon=True)
    t.start()
    
    ret = app.exec_()
    lock_file.unlock() # Release lock on exit
    sys.exit(ret)

if __name__ == "__main__":
    main()
