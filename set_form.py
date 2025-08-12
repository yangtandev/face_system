from ui.setui import Ui_Form
from setting.ip_set import set_ip

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication

import signal

import os, sys, json

main_path = os.path.dirname(__file__)

signal.signal(signal.SIGINT, signal.SIG_DFL)

json_path = os.path.join(os.path.dirname(__file__), "config.json")
config_ = {}
try:
    with open(json_path, "r", encoding="utf-8") as json_file:
        config_ = json.load(json_file)
except Exception as e:
    print("載入失敗", e)

class MainWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.send.clicked.connect(self.save)
        self.btn_ip.clicked.connect(self.sena_ip)
        self.clothes.setChecked(config_["Clothes_detection"])
        self.clothes_2.setChecked(config_["Clothes_show"])
        self.debug.setChecked(config_["test_mod"])
        self.auto_open.setChecked(config_["auto_open"])
        self.full_screen.setChecked(config_["full_screen"])

        self.camera_in.setPlainText(config_["cameraIP"]["in_camera"])
        self.camera_out.setPlainText(config_["cameraIP"]["out_camera"])
        try:
            self.ec2_ip.setPlainText(config_["Server"]["ip"])
            self.ec2_name.setPlainText(config_["Server"]["username"])
            self.ec2_pass.setPlainText(config_["Server"]["password"])
            self.ec2_project.setPlainText(config_["Server"]["API_url"])
            self.ec2_ID.setPlainText(str(config_["Server"]["location_ID"]))
            dir_ = config_["Server"]["face_data_dir"].split("/")[-2]
            self.ec2_dir.setPlainText(dir_)
        except:
            pass
        self.min_face.setPlainText(str(config_["min_face"]))

        self.in_voice.setPlainText(config_["say"]['in'])
        self.out_voice.setPlainText(config_["say"]['out'])
        self.clothes_voice.setPlainText(config_["say"]['clothes'])

        self.loc_ip.setPlainText(config_["ip_set"]["ip_address"])
        self.mask_ip.setPlainText(config_["ip_set"]["ip_mask"])
        self.gateway_ip.setPlainText(config_["ip_set"]["ip_gateway"])

        self.door.setPlainText(str(config_["door"]))

    def save(self):
        copy_config = config_.copy()
        copy_config["cameraIP"]["in_camera"] = self.camera_in.toPlainText()
        copy_config["cameraIP"]["out_camera"] = self.camera_out.toPlainText()

        copy_config["Clothes_detection"] = self.clothes.isChecked()
        copy_config["Clothes_show"] = self.clothes_2.isChecked()
        copy_config["test_mod"] = self.debug.isChecked()
        copy_config["auto_open"] = self.auto_open.isChecked()
        copy_config["full_screen"] = self.full_screen.isChecked()

        copy_config["Server"]["ip"] = self.ec2_ip.toPlainText()
        copy_config["Server"]["username"] = self.ec2_name.toPlainText()
        copy_config["Server"]["password"] = self.ec2_pass.toPlainText()
        copy_config["Server"]["API_url"] = self.ec2_project.toPlainText()
        copy_config["Server"]["location_ID"] = int(self.ec2_ID.toPlainText())
        copy_config["Server"]["face_data_dir"] = f"/home/{self.ec2_name.toPlainText()}/{self.ec2_dir.toPlainText()}/media"

        copy_config["min_face"] = int(self.min_face.toPlainText())

        copy_config["say"]['in'] = self.in_voice.toPlainText()
        copy_config["say"]['out'] = self.out_voice.toPlainText()
        copy_config["say"]['clothes'] = self.clothes_voice.toPlainText()

        copy_config["door"] = self.door.toPlainText()
        mbox = QtWidgets.QMessageBox(self)       # 加入對話視窗
        ret = mbox.question(self, '提問', '是否儲存目前設置的參數?')
        if ret == QtWidgets.QMessageBox.Yes:
            try:
                with open(json_path , "w") as fp:
                    json.dump(copy_config, fp, indent=2)
                    mbox.information(self, '通知', '儲存成功，請重新啟動人臉辨識主程式')
            except Exception as e:
                mbox.warning(self, '警告', f'儲存失敗\n{e}')
        elif ret == QtWidgets.QMessageBox.No:
            pass
        elif ret == QtWidgets.QMessageBox.Cancel:
            pass
        pass

    def sena_ip(self):
        copy_config = config_.copy()
        copy_config["ip_set"]["ip_address"] = self.loc_ip.toPlainText()
        copy_config["ip_set"]["ip_mask"] = self.mask_ip.toPlainText()
        copy_config["ip_set"]["ip_gateway"] = self.gateway_ip.toPlainText()

        mbox = QtWidgets.QMessageBox(self)       # 加入對話視窗
        ret = mbox.question(self, '提問', '是否修改IP?')
        if ret == QtWidgets.QMessageBox.Yes:
            try:
                re = set_ip(copy_config["ip_set"]["ip_address"],
                       copy_config["ip_set"]["ip_mask"],
                       copy_config["ip_set"]["ip_gateway"])
                if int(re) != 0:
                    mbox.warning(self, '警告', f'儲存失敗\n{e}')
                else:
                    with open(json_path , "w") as fp:
                        json.dump(copy_config, fp, indent=2)
                        mbox.information(self, '通知', 'IP修改完成')
            except Exception as e:
                mbox.warning(self, '警告', f'儲存失敗\n{e}')
        elif ret == QtWidgets.QMessageBox.No:
            pass
        elif ret == QtWidgets.QMessageBox.Cancel:
            pass
        pass

if __name__ == "__main__":

    app = QApplication(sys.argv)
    desktop = QApplication.desktop()
    form = MainWindow()
    form.setWindowTitle(f"參數設定視窗")

    form.show()
    sys.exit(app.exec_())
