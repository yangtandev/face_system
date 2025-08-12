from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":     
    app = QApplication(sys.argv)
    desktop = QApplication.desktop()
    screen_count = desktop.screenCount() #讀取螢幕數量
    print("螢幕數量:", screen_count)
    for i in range(screen_count):
        print(desktop.screenGeometry(i).width())
