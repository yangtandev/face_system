from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from ui.user import Ui_Show_from
from ui.dy_user import Ui_dynamic_Form
from ui.user_only import Ui_Form
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication
import os, json, time

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
        # 設定定時器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_screen)
        self.timer.start(30000)  # 每1000毫秒（1秒）更新一次

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
        
