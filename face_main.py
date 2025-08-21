from pathlib import Path
import time, threading, queue, json, os, subprocess
import sys

from models import mtcnn, inception_resnet_v1
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import requests
from tqdm import tqdm
from ultralytics import YOLOv10
from PVMS_Library import config
from init.log import LOGGER
from ui.user_show import MainWindow
from py_ssh import ssh
from init.say import Say_
#import function as fun
import datetime
from init.camera import VideoCapture
from init.model import Detector, Comparison
from function import *

from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets

from sklearn.svm import LinearSVC
import pickle
import signal

from dataclasses import dataclass
from typing import List, Dict, Any

main_path = os.path.dirname(__file__)

# def exit_all(a, b):
#     for i in range(len(camera)):
#         camera[i].terminate()
#     os._exit(0)

def check_empty_string_in_dict(data):
    for key, value in data.items():
        if isinstance(value, dict):
            # 如果值是字典，递归调用
            if not check_empty_string_in_dict(value):
                return False
        elif value == "":
            # 如果值为空字符串，返回False
            return False
    return True

# signal.signal(signal.SIGINT, exit_all)

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

font_path = os.path.join(os.path.dirname(__file__), "other/NotoSansTC-VariableFont_wght.ttf")
CAMERA = {0:"inCamera", 1:"outCamera"}


def put_chinese_text(img, text, position, font_path, font_size, color, background=True):
    img_pil = Image.fromarray(img)#Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox(position, text, font=font)
    if background:
        draw.rectangle(text_bbox, fill='white')
    draw.text(position, text, font=font, fill=color)
    img = np.array(img_pil)# cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    del img_pil
    return img

class CameraSystem:
    """
    負責處理個別攝影機影像流程：
    - 僅負責根據 Comparison 模組提供的狀態來繪製畫面。
    - 控制聲音播報與 UI 更新。
    """
    def __init__(self, ip, frame_num, n, system):
        self.system = system
        self.camera = VideoCapture(ip)
        self.frame_num = frame_num
        self.stop_threads = False
        self.show_frame = np.array([])
        self.image = cv2.imread(os.path.join(os.path.dirname(__file__), "other/mask.png"))
        if CONFIG[CAMERA[frame_num]]["close"]:
            width = self.image.shape[1]
            self.image = self.image[0:, width // 5 : 4 * width // 5]
        self.speak_time = 0
        self.save_img_time = [0, 0]
        self.save_name_last = ""
        self.clothes_de = (CONFIG["Clothes_show"] and self.frame_num == 0)

        self.detect = Detector(frame_num, system)
        self.compar = Comparison(frame_num, system)

        threading.Thread(target=self.main_camera, daemon=True).start()

        self.n_camera = n < 2
        if frame_num == 1 and n < 2 :
            return

        self.win = MainWindow(self.updata_screen, frame_num)
        self.img1_size = (self.win.img1.width(), self.win.img1.height())
        self.img2_size = (self.win.img2.width(), self.win.img2.height())

        if n == 1:
            self.win.setWindowTitle(f"進出視窗")

        self.win.closeEvent = self.terminate
        if self.frame_num == 0 and CONFIG["Clothes_show"]:
            self.win.img3.setPixmap(QPixmap(f'{main_path}/other/helmet_R.png'))
            self.win.img4.setPixmap(QPixmap(f'{main_path}/other/vest_R.png'))
            self.win.img3.setStyleSheet("QLabel{background-color: rgba(255,0,0,255);}")
            self.win.img4.setStyleSheet("QLabel{background-color: rgba(255,0,0,255);}")
            self.win.img3.setScaledContents(True)
            self.win.img4.setScaledContents(True)

    def main_camera(self):
        """
        主迴圈持續處理畫面並根據狀態繪製 UI 元素。
        """
        while not self.stop_threads:
            self.system.state.frame[self.frame_num] = self.camera.read()
            now_frame = self.system.state.frame[self.frame_num].copy()
            font_size = 60

            # 繪製人臉框
            if self.system.state.max_box[self.frame_num] is not None:
                x1, y1, x2, y2 = self.system.state.max_box[self.frame_num]
                cv2.rectangle(now_frame, (x1, y1), (x2, y2), (255, 0, 0), 6)

                # 繪製歷史紀錄
                history_lines = self.system.state.display_history[self.frame_num]
                if history_lines:
                    for i, line in enumerate(history_lines):
                        y_offset = y1 + (i * (font_size // 2 + 5))
                        now_frame = put_chinese_text(now_frame, line, (x2 + 10, y_offset), font_path, font_size // 2, (255, 255, 0), background=True)

                # 根據 same_class 狀態決定顯示文字
                current_class = self.system.state.same_class[self.frame_num]
                if current_class != "None":
                    staff_name = self.system.state.features_dict.get("id_name", {}).get(current_class, "訪客")
                    now_frame = put_chinese_text(now_frame, staff_name, (x1, y1-55), font_path, font_size, (205,0,0))
                else:
                    now_frame = put_chinese_text(now_frame, "辨識中", (x1, y1-55), font_path, font_size, (0,0,0))

                # 觸發簽到/簽離 (API 呼叫)
                if self.system.state.same_people[self.frame_num] >= 1:
                    success_staff_name = self.system.state.features_dict.get("id_name", {}).get(current_class, "訪客")

                    if current_class != "None" and (not CONFIG["Clothes_detection"] or (self.system.state.clothes[0] and self.system.state.clothes[2])):
                        check_in_out(self.system, success_staff_name, current_class, self.frame_num, self.n_camera)

                    self.save_img(self.system.state.frame[self.frame_num], "face", success_staff_name)
                    # 重設觸發器，防止重複呼叫
                    self.system.state.same_people[self.frame_num] = 0

            self.show_frame = now_frame
        LOGGER.info("main_camera, 已跳出迴圈")

    def updata_screen(self):
        time.sleep(0.5)
        self.win.my_thread.signal_update_img.connect(self.win.update_img)
        self.win.my_thread.signal_update_hint.connect(self.win.update_hint)
        if self.clothes_de:
            self.win.my_thread.signal_update_bgcolor.connect(self.win.update_bgcolor)
        while not self.stop_threads:
            if self.show_frame.shape[0] == 0:
                continue
            try:
                self.win.my_thread.signal_update_img.emit(self.win.img1, self.show_main())
                self.win.my_thread.signal_update_img.emit(self.win.img2, self.shwo_head())
                if self.clothes_de:
                    img3, img4 = self.show_save()
                    self.win.my_thread.signal_update_img.emit(self.win.img3, img3)
                    self.win.my_thread.signal_update_img.emit(self.win.img4, img4)
                color, txt = self.show_hint()
                self.win.my_thread.signal_update_hint.emit(self.win.hint, color, txt)
            except BaseException as e:
                if self.clothes_de and CONFIG["test_mod"]:
                    print("updata_screen", e)
                pass
            time.sleep(0.00001)
        LOGGER.info("updata_screen, 已跳出迴圈")

    def show_main(self):
        alpha = 0.1
        show_img = self.show_frame
        height, width, _ = show_img.shape
        self.image = cv2.resize(self.image, (width, height))
        img1_size = (self.img1_size[0], self.img1_size[1])
        show_img = cv2.addWeighted(self.image, alpha, self.show_frame, 1 - alpha, 0)
        show_img = cv2.resize(show_img, img1_size)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        height, width, channel = show_img.shape
        bytesPerline = channel * width
        img = QImage(show_img, width, height, bytesPerline, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    def shwo_head(self):
        path = f'{main_path}/other/clear_img.png'
        if self.system.state.same_class[self.frame_num] != "None":
            path = self.system.state.profile_dict.get(self.system.state.same_class[self.frame_num], path)
        img = QPixmap(path)
        return img

    def show_save(self):
        helmet = "helmet_R.png"
        vest = "vest_R.png"
        if self.system.state.clothes[0]:
            vest = "vest_G.png"
        if self.system.state.clothes[2]:
            helmet = "helmet_G.png"
        img3 = QPixmap(f'{main_path}/other/{helmet}')
        img4 = QPixmap(f'{main_path}/other/{vest}')
        return img3, img4

    def show_hint(self):
        color = ""
        txt = ""
        current_class = self.system.state.same_class[self.frame_num]
        if current_class != "None":
            color = 'color: rgb(0, 170, 0); background-color: rgb(255, 255, 255); font: 24pt "微軟正黑體";'
            txt = self.system.state.features_dict.get("id_name", {}).get(current_class, "訪客")
        else:
            color = 'color: rgb(0, 85, 255); background-color: rgb(255, 255, 255); font: 24pt "微軟正黑體";'
            txt = "辨識中"
        return color, txt

    def save_img(self, img, path, staffname=""):
        dict_={"face":0, "clothes":1}
        img = cv2.resize(img, (800, 600))
        datetime_dt = datetime.datetime.today()
        date_str = datetime_dt.strftime("%Y_%m_%d")
        time_str = datetime_dt.strftime("%H;%M;%S")
        os.makedirs(f"{main_path}/img_log/{path}/{date_str}", exist_ok=True)
        if time.time()-self.save_img_time[dict_[path]] > 5 or (self.save_name_last != staffname and staffname != ""):
            cv2.imwrite(f"{main_path}/img_log/{path}/{date_str}/{time_str}_{staffname}.jpg", img)
            self.save_img_time[dict_[path]] = time.time()
            if staffname != "":
                self.save_name_last = staffname

    def terminate(self, k=None):
        self.stop_threads = True
        if hasattr(self, 'win'):
            self.win.my_thread.exit()
        self.camera.terminate()
        self.detect.terminate()
        self.compar.terminate()

#-------------------------------------
# Load configuration

with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as json_file:
    CONFIG = json.load(json_file)

if not check_empty_string_in_dict(CONFIG):
    LOGGER.info("設定有缺漏關閉程式")
    print("設定有缺漏關閉程式")
    os._exit(0)

API = config.API(str(CONFIG["Server"]["API_url"]), int(CONFIG["Server"]["location_ID"]))

@dataclass
class GlobalState:
    max_box: List[Any] = None
    same_people: List[int] = None
    same_class: List[str] = None
    frame: List[Any] = None
    frame_mtcnn: List[Any] = None
    clothes: List[bool] = None
    check_time: Dict[str, List[Any]] = None
    features_dict: Dict[str, Any] = None
    profile_dict: Dict[str, str] = None
    display_history: List[list] = None # 用於在畫面上顯示歷史紀錄
    leave: int = 0
    min_face: int = 0

class FaceRecognitionSystem:
    """
    主系統控制類別，負責：
    - 建立模型與狀態管理結構
    - 同步人臉資料與向量
    - 建立並控制雙鏡頭人臉與裝備辨識流程
    - 執行 PyQt5 應用主循環
    """
    def __init__(self):
        """
        初始化全域狀態、模型與參數設定。
        """
        self.state = GlobalState()
        self.state.max_box = [None, None]
        self.state.same_people = [0, 0]
        self.state.same_class = ["None", "None"]
        self.state.frame = [None, None]
        self.state.frame_mtcnn = [None, None]
        self.state.clothes = [False, False, False]
        self.state.check_time = {}
        self.state.features_dict = {}
        self.state.profile_dict = {}
        self.state.display_history = [[], []]
        self.state.min_face = CONFIG["min_face"]

        self.svc = LinearSVC(C=1, multi_class='ovr')
        self.mtcnn = mtcnn.MTCNN(image_size=160, min_face_size=150, keep_all=True, select_largest=True)
        self.resnet = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2').eval()

        if CONFIG["Clothes_show"]:
            print("建立服裝辨識模型")
            LOGGER.info("建立服裝辨識模型")
            models_dir = Path(f'{os.path.dirname(__file__)}/models')
            model_name = 'best_cloth2'
            int8_model_det_path = models_dir/'int8'/f'{model_name}_openvino_model/{model_name}.xml'
            self.model_clothes = YOLOv10(int8_model_det_path.parent, task='detect')

        self.speaker = Say_()
        self.local_media_path = os.path.join(os.path.dirname(__file__), "media")
        self.model_path = os.path.join(self.local_media_path, "linear_svc_model.pkl")

        # 確保所有必要的 media 子目錄都存在
        os.makedirs(os.path.join(self.local_media_path, "descriptors"), exist_ok=True)
        os.makedirs(os.path.join(self.local_media_path, "pic_bak"), exist_ok=True)
        os.makedirs(os.path.join(self.local_media_path, "profile_pictures"), exist_ok=True)

        self.update_lock = threading.Lock() # 用於防止多個更新執行緒同時運行

    def run(self):
        """
        啟動系統主流程：
        - 立即載入本地現有模型開始辨識。
        - 在背景啟動一個執行緒來執行首次的資料同步與更新。
        - 建立 UI 與攝影機視窗。
        - 進入 PyQt 事件主迴圈。
        """
        print("系統啟動中...")
        # 1. 嘗試載入本地現有模型，讓系統可以立即開始服務
        self._load_svc_model()
        self._load_features_and_profiles()

        # 2. 在背景啟動資料更新流程
        print("在背景啟動首次資料同步...")
        threading.Thread(target=self.update_data_and_model, daemon=True).start()

        # 3. 啟動UI和攝影機
        self.update_inout_log()
        app = QApplication(sys.argv)
        self.setup_cameras()
        sys.exit(app.exec_())

    def update_data_and_model(self, initial_run=True):
        """
        資料更新與模型訓練的主流程，設計為可在背景執行緒中安全運行。
        """
        if not self.update_lock.acquire(blocking=False):
            print("更新流程已在運行中，跳過本次觸發。")
            return

        try:
            print("\n===== 開始背景資料更新流程 =====")
            # 步驟 1: 與遠端伺服器同步檔案
            sync_success = self._sync_files_with_server()
            if not sync_success:
                print("檔案同步失敗，中止本次更新。")
                return

            # 步驟 2: 找出新增和刪除的圖片
            pic_bak_path = os.path.join(self.local_media_path, "pic_bak")
            descriptors_path = os.path.join(self.local_media_path, "descriptors")

            local_pics = set(f.split('.')[0] for f in os.listdir(pic_bak_path))
            local_descriptors = set(f.split('.')[0] for f in os.listdir(descriptors_path))

            new_files = [f"{name}.jpg" for name in local_pics - local_descriptors] # 假設都是 jpg
            deleted_files = [f"{name}.npy" for name in local_descriptors - local_pics]

            update_needed = bool(new_files or deleted_files)
            print(f"檔案比對完成：新增 {len(new_files)} 張圖片，刪除 {len(deleted_files)} 個特徵檔。")

            # 步驟 3: 處理檔案變動
            if update_needed:
                self._process_deleted_descriptors(deleted_files, descriptors_path)
                self._generate_new_descriptors(new_files, pic_bak_path, descriptors_path)

                # 步驟 4: 重新載入特徵並訓練模型
                features, X_train, y_train, X_test, y_test = self._load_features_from_disk()
                self._train_and_save_svc_model(X_train, y_train, X_test, y_test)

                # 步驟 5: 熱更新系統狀態
                self.state.features_dict = features
                self._update_profile_pictures()
                print("模型與資料已熱更新完畢。")
            else:
                print("資料無變動，無需更新模型。")

        finally:
            self.update_lock.release()
            # 如果不是初次運行，則設定下一次的定時更新
            if not initial_run:
                print(f"===== 背景資料更新流程結束，將在 300 秒後再次執行 =====")

            # 設定下一次定時器
            threading.Timer(300, self.update_data_and_model, args=[False]).start()

    def _sync_files_with_server(self):
        """與遠端伺服器同步檔案"""
        server_dir = CONFIG["Server"]["face_data_dir"]
        ssh_config = {"ip": CONFIG["Server"]["ip"], "username": CONFIG["Server"]["username"]}

        print("同步人員照片 (pic_bak)...")
        source_pic_bak = os.path.join(server_dir, "pic_bak").replace('\\', '/')
        dest_pic_bak = os.path.join(self.local_media_path, "pic_bak").replace('\\', '/')
        if not ssh.sync_with_rsync(ssh_config, source_pic_bak, dest_pic_bak):
            return False

        print("同步大頭照 (profile_pictures)...")
        source_profile = os.path.join(server_dir, "profile_pictures").replace('\\', '/')
        dest_profile = os.path.join(self.local_media_path, "profile_pictures").replace('\\', '/')
        if not ssh.sync_with_rsync(ssh_config, source_profile, dest_profile):
            return False

        return True

    def _process_deleted_descriptors(self, deleted_files, descriptors_path):
        """刪除無效的特徵檔"""
        if not deleted_files: return
        print(f"正在刪除 {len(deleted_files)} 個過期的特徵檔...")
        for f in deleted_files:
            try:
                os.remove(os.path.join(descriptors_path, f))
            except OSError as e:
                print(f"刪除檔案 {f} 失敗: {e}")

    def _generate_new_descriptors(self, new_files, pic_bak_path, descriptors_path):
        """為新圖片產生特徵檔"""
        if not new_files: return
        print(f"正在為 {len(new_files)} 張新圖片產生特徵檔...")
        for filename in tqdm(new_files):
            image_path = os.path.join(pic_bak_path, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                img_cropped = self.mtcnn(image)
                if img_cropped is not None:
                    img_embedding = self.resnet(img_cropped)
                    np.save(os.path.join(descriptors_path, f"{filename.split('.')[0]}.npy"), img_embedding[0].detach().numpy())
                else:
                    print(f"警告：在圖片 {filename} 中未偵測到人臉。")
            except Exception as e:
                print(f"處理圖片 {filename} 時發生錯誤: {e}")

    def _load_features_from_disk(self):
        """從硬碟載入所有特徵檔，並劃分訓練/測試集"""
        print("從硬碟載入所有特徵檔...")
        descriptors_path = os.path.join(self.local_media_path, "descriptors")
        X_train, y_train, X_test, y_test = [], [], [], []
        features = {"id_name": {}}

        all_descriptors = [f for f in os.listdir(descriptors_path) if f.lower().endswith('.npy')]
        for filename in all_descriptors:
            path = os.path.join(descriptors_path, filename)
            category = filename.split("_")[0]
            name = filename.split("_")[-1].split(".")[0]
            load = np.load(path)

            if category not in features:
                features[category] = []
                X_test.append(load)
                y_test.append(category)
            else:
                X_train.append(load)
                y_train.append(category)

            features[category].append(load)
            features["id_name"][category] = name

        print(f"特徵檔載入完成。訓練集: {len(X_train)}, 測試集: {len(X_test)}")
        return features, X_train, y_train, X_test, y_test

    def _train_and_save_svc_model(self, X_train, y_train, X_test, y_test):
        """訓練 SVC 模型並儲存"""
        if not X_train or not y_train:
            print("警告：訓練集為空，無法訓練模型。")
            return

        print("開始訓練 SVC 模型...")
        self.svc.fit(X_train, y_train)
        print("模型訓練完畢。")

        if X_test and y_test:
            score = self.svc.score(X_test, y_test)
            print(f"模型驗證分數 (Accuracy): {score:.2%}")

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.svc, f)
        print(f"模型已儲存到: {self.model_path}")

    def _load_svc_model(self):
        """載入本地現有的 SVC 模型"""
        if os.path.exists(self.model_path):
            print(f"正在從 {self.model_path} 載入現有模型...")
            try:
                with open(self.model_path, 'rb') as f:
                    self.svc = pickle.load(f)
                print("模型載入成功。")
            except Exception as e:
                print(f"載入模型失敗: {e}")
        else:
            print("本地找不到現有模型，將在首次同步後建立。")

    def _load_features_and_profiles(self):
        """載入本地現有的特徵和頭像資料"""
        try:
            features, _, _, _, _ = self._load_features_from_disk()
            self.state.features_dict = features
            self._update_profile_pictures()
            print("本地特徵與頭像資料載入完成。")
        except Exception as e:
            print(f"載入本地特徵時發生錯誤: {e}")

    def _update_profile_pictures(self):
        """更新大頭照字典"""
        profile_path = os.path.join(self.local_media_path, "profile_pictures")
        if not os.path.isdir(profile_path): return

        copy_profile = {}
        for filename in os.listdir(profile_path):
            category = filename.split("_")[0]
            if category not in copy_profile:
                copy_profile[category] = os.path.join(profile_path, filename)
        self.state.profile_dict = copy_profile

    # setup_cameras, update_inout_log, shutdown 等方法維持不變
    def setup_cameras(self):
        """
        建立並初始化每個攝影機對應的 CameraSystem 實例。
        若兩鏡頭 IP 相同則僅建立一個視窗。
        """
        ips = [CONFIG["cameraIP"]["in_camera"], CONFIG["cameraIP"]["out_camera"]]
        n = 2 if ips[0] != ips[1] else 1
        self.cameras = []
        for i, ip in enumerate(ips):
            if ip != "0":
                self.cameras.append(CameraSystem(ip, i, n, self))

        for i in range(len(self.cameras)-(2-n)):
            self.cameras[i].win.show()

    def update_inout_log(self):
        """
        每 5 分鐘向伺服器請求已簽到名單，更新 check_time 狀態,
        以便辨識時確認進/出狀態是否重複。
        """
        datetime_dt = datetime.datetime.today()# 獲得當地時間
        date_str = datetime_dt.strftime("%m/%d %H:%M:%S")  # 格式化日期
        try:
            log_json = API.Scan_today_log()
            for staff_id in log_json.keys():
                #解析json
                if log_json[staff_id]["state"] == "enter":
                    self.state.check_time[staff_id] = [False, time.time()-100]
                elif staff_id in self.state.check_time.keys():
                    self.state.check_time[staff_id] = [True, 0]
                    threading.Timer(5,clear_leave_employee, (self, staff_id, )).start()
            print(date_str, "更新已簽到名單")
            LOGGER.info("更新已簽到名單")

        except Exception as e:
            LOGGER.info("已簽到名單更新失敗"+str(e))
            print(date_str, "已簽到名單更新失敗", e)
        updata_log_timer = threading.Timer(300, self.update_inout_log)
        updata_log_timer.start()

    def shutdown(self):
        """
        安全地關閉所有攝影機執行緒。
        """
        print("\nShutting down... Please wait.")
        if hasattr(self, 'cameras'):
            for cam in self.cameras:
                cam.terminate()
        # 等待一小段時間確保執行緒都已收到終止信號
        time.sleep(1)
        os._exit(0)

if __name__ == "__main__":
    face_recognition_system = FaceRecognitionSystem()
    # face_recognition_system.run()

    # 建立系統實例
    face_recognition_system = FaceRecognitionSystem()

    # 定義一個新的訊號處理函式，讓它可以存取到 face_recognition_system 實例
    def signal_handler(sig, frame):
        face_recognition_system.shutdown()

    # 將 SIGINT (Ctrl+C) 訊號綁定到新的處理函式
    signal.signal(signal.SIGINT, signal_handler)

    # 啟動主程式
    try:
        face_recognition_system.run()
    except SystemExit:
        # 捕捉由 app.exec_() 關閉時可能引發的 SystemExit
        face_recognition_system.shutdown()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        face_recognition_system.shutdown()
