# 匯入 UI 主畫面類別（由 Qt Designer 轉出的 .py）
from ui.main import Ui_Form

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtGui import QImage, QPixmap

import signal
import os, sys, json, threading

from setting.bulid_config import update_config_file

# === 全域主程式路徑 ===
main_path = os.path.dirname(__file__)

# === 啟用 Ctrl+C 安全中斷應用程式（重要） ===
signal.signal(signal.SIGINT, signal.SIG_DFL)

# === 若 config 檔不存在則建立/更新 ===
update_config_file(os.path.join(main_path, "config.json"))

# === 讀取 config 設定檔 ===
json_path = os.path.join(main_path, "config.json")
config_ = {}
try:
    with open(json_path, "r", encoding="utf-8") as json_file:
        config_ = json.load(json_file)
except Exception as e:
    print("載入失敗", e)


class MainWindow(QWidget, Ui_Form):
    """
    主選單介面視窗，繼承自 PyQt5 QWidget 與自定義的 Ui_Form。

    功能按鈕：
    - pushButton     => 開啟設定介面
    - pushButton_2   => 啟動人臉系統
    """

    def __init__(self, parent=None):
        """
        初始化主視窗，建立 UI 元件並綁定按鈕事件。

        Parameters:
        parent (QWidget or None): 父層視窗
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.open_set)
        self.pushButton_2.clicked.connect(self.open_faceSys)

        
    def open_set(self):
        """
        使用 bash 執行人臉系統主程式 run_face_system.sh。
        """
        os.system(f"bash {main_path}/run_set_form.sh")

    def open_faceSys(self):
        os.system(f"bash {main_path}/run_face_system.sh")
    

if __name__ == "__main__":     
    
    app = QApplication(sys.argv)
    desktop = QApplication.desktop()
    form = MainWindow()
    form.setWindowTitle(f"選單視窗")

    if config_["auto_open"]:
        th = threading.Thread(target=os.system(f"bash {main_path}/run_face_system.sh"))
        th.daemon = True             
        th.start()
    os.system(f'echo "$LOGNAME"')
    form.show()
    
    sys.exit(app.exec_())