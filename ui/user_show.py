from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from ui.user import Ui_Show_from
from ui.dy_user import Ui_dynamic_Form
from ui.user_only import Ui_Form
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QPushButton, QInputDialog, QLineEdit, QMessageBox
from PyQt5 import QtCore
import os, json, time, subprocess, sys

config_ = {}
try:
    with open(os.path.join(os.path.dirname(__file__), "../config.json"), "r", encoding="utf-8") as json_file:
        config_ = json.load(json_file)
except Exception as e:
    print("ui_show-載入失敗", e)

class MainWindow(QWidget, Ui_Form):
    def __init__(self, fun, frame_num, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "face-detection.png")))
        self.resizeEvent = self.win_resize
        self.obj = [self, self.img1, self.hint2, self.img2, self.img3, self.img4, self.hint]
        
        # [2026-01-30 Fix] Always reload config on init to support soft reload
        self.reload_config()
        
        # [2026-01-30 Fix] Clear hardcoded styles (white bg) from auto-generated UI to allow Dark Theme
        try:
            # user_only.py hardcodes white backgrounds on these labels
            self.img1.setStyleSheet("")
            self.hint.setStyleSheet("")
            self.hint2.setStyleSheet("")
            # Also clear any others if inherited from other UIs (though user_show uses user_only)
            self.min_face.setStyleSheet("")
            self.in_voice.setStyleSheet("")
            self.out_voice.setStyleSheet("")
            self.clothes_voice.setStyleSheet("")
        except Exception: pass
        
        if frame_num == 0:
            self.setWindowTitle(f"進入視窗")
        if frame_num == 1:
            self.setWindowTitle(f"離開視窗")

        self.img1.setScaledContents(True)
        self.img2.setScaledContents(True)
        self.img3.setStyleSheet("QLabel{background-color: rgba(255,255,255,0);}")
        self.img4.setStyleSheet("QLabel{background-color: rgba(255,255,255,0);}")

        self.org_point = []
        for i in range( len(self.obj)):
            height = self.obj[i].geometry().height()
            left_ = self.obj[i].geometry().left()
            width = self.obj[i].geometry().width()
            top = self.obj[i].geometry().top()
            self.org_point.append([height, left_, width, top])
        
        self.frame_num = frame_num
        self.my_thread = MyThread()
        self.my_thread.run = fun
        self.my_thread.start()
        self.update_screen()
        
        # [2026-01-30 Feature] Add Settings Button
        self.btn_setting = QPushButton("⚙", self)
        self.btn_setting.setGeometry(10, 10, 40, 40)
        self.btn_setting.setStyleSheet("background-color: rgba(0,0,0,100); color: white; border-radius: 5px; font-size: 20px;")
        self.btn_setting.clicked.connect(self.open_settings)
        self.btn_setting.show()
        self.btn_setting.raise_()

        # 設定定時器

    def reload_config(self):
        global config_
        try:
            with open(os.path.join(os.path.dirname(__file__), "../config.json"), "r", encoding="utf-8") as json_file:
                config_ = json.load(json_file)
        except Exception as e:
            print("ui_show-重新載入失敗", e)
        # [2026-01-19 Fix] 移除 30 秒自動重置視窗大小的機制，允許使用者手動調整版面
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_screen)
        # self.timer.start(30000)  # 每1000毫秒（1秒）更新一次

    def open_settings(self):
        """Open the external setting tool with password protection."""
        # [2026-01-30 Fix] Use explicit QInputDialog to ensure centering
        dlg = QInputDialog(self)
        dlg.setWindowTitle('身分驗證')
        dlg.setLabelText('請輸入管理員密碼:')
        dlg.setTextEchoMode(QLineEdit.Password)
        
        # Force center on parent
        # Note: dlg.exec_() blocks, so we move before exec.
        # But dlg size might not be calculated yet.
        # We trust Qt parent centering usually, but if it fails (top-left),
        # we can try to force move.
        
        if dlg.exec_() == QInputDialog.Accepted:
            text = dlg.textValue()
            # Default password is 'admin', or matching the server password if available?
            # Let's use 'admin' for simplicity as requested "Option A".
            if text == 'admin':
                try:
                    # Launch setting_tool.py as a separate process
                    tool_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "setting_tool.py")
                    
                    # [2026-01-30 Fix] Calculate global geometry for correct centering (even in fullscreen)
                    global_pos = self.mapToGlobal(QtCore.QPoint(0, 0))
                    
                    args = [
                        sys.executable, tool_path,
                        "--parent_x", str(global_pos.x()),
                        "--parent_y", str(global_pos.y()),
                        "--parent_w", str(self.width()),
                        "--parent_h", str(self.height())
                    ]
                    
                    subprocess.Popen(args)
                except Exception as e:
                    QMessageBox.critical(self, "錯誤", f"無法啟動設定工具: {e}")
            else:
                QMessageBox.warning(self, "錯誤", "密碼錯誤")

    def update_img( self, obj, pixmap:QPixmap):
        obj.setPixmap(pixmap)

    def update_bgcolor(self, obj, color):
        for i in range(len(obj)):
            obj[i].setStyleSheet(color[i])

    def update_hint(self, obj, color, txt):
        obj.setStyleSheet(color)
        obj.setText(txt)

    def win_resize(self, event):
        Proportion_X = self.width()/self.org_point[0][2]
        Proportion_Y = self.height()/self.org_point[0][0]
        blank_X = 0
        blank_Y = 0
        
        chang = min(Proportion_X, Proportion_Y)
        for i in range(1, len(self.obj)):
            height = self.org_point[i][0]*chang
            left_ = self.org_point[i][1]*chang
            width = self.org_point[i][2]*chang
            top = self.org_point[i][3]*chang
            if i == 1:
                height_hint2 = self.org_point[2][0]*chang
                blank_X = max(0, (self.width() - width )//2)
                blank_Y = max(0, (self.height() - height - height_hint2)//2)
        
            self.obj[i].setGeometry(int(left_+blank_X), int(top+blank_Y),  int(width), int(height))
        pass

    def update_screen(self):
        desktop = QApplication.desktop()
        screen_count = desktop.screenCount() #讀取螢幕數量
        n = 2
        if config_["cameraIP"]["in_camera"] == config_["cameraIP"]["out_camera"]:
            n = 1
        elif config_["cameraIP"]["in_camera"] == "0" or config_["cameraIP"]["out_camera"] == "0":
            n = 1

        # [2026-01-19] 強制視窗排版邏輯 (當 full_screen = False)
        # 即使只有單螢幕，也強制將視窗左右並排，方便測試與除錯
        if not config_.get("full_screen", True):
            # [2026-01-19 Fix] 改用 availableGeometry 取得扣除系統工具列(Dock)後的可用區域
            # 避免視窗被 Ubuntu 左側/底部工具列遮擋
            avail_rect = desktop.availableGeometry(0)
            x_offset = avail_rect.x()
            y_offset = avail_rect.y()
            w = avail_rect.width()
            h = avail_rect.height()
            
            # 若只有單一視窗 (n=1)，預設佔據可用區域的左半邊
            if n == 1:
                 self.setGeometry(x_offset, y_offset, w // 2, h)
            else:
                # 雙視窗模式：Cam 0 (Entry) 左邊，Cam 1 (Exit) 右邊
                if self.frame_num == 0:
                    self.setGeometry(x_offset, y_offset, w // 2, h)
                elif self.frame_num == 1:
                    # 注意：起始 X 座標必須加上左半邊的寬度
                    self.setGeometry(x_offset + (w // 2), y_offset, w // 2, h)
            
            self.showNormal()
            return

        if n == screen_count:
            pass

        if screen_count > 1:
            rect = desktop.screenGeometry(self.frame_num)
            self.move(rect.left(), rect.top())
            self.resize(rect.width()//2, rect.height())
        else:
            helf_w = desktop.screenGeometry(0).width()
            helf_h = desktop.screenGeometry(0).height()
            if helf_h > helf_w:
                self.move(0, self.frame_num*helf_h//2)
                self.resize(helf_w, helf_h//2)
            else:
                self.move(self.frame_num*helf_w//2, 0)
                self.resize(helf_w//2, helf_h)

        if config_["full_screen"] and n == screen_count:
            self.showMaximized()

class MyThread(QThread):
    signal_update_img = pyqtSignal(QLabel, QPixmap)
    signal_update_bgcolor = pyqtSignal(list, list)
    signal_update_hint = pyqtSignal(QLabel, str, str)

    def __init__(self):
        super(MyThread, self).__init__()

    def run(self):
        pass
        """ while True:
            #print(current_time)
            #放置參數更新涵式
            time.sleep(0.001) """
        
