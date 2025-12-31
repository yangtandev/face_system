from pathlib import Path
import time, threading, queue, json, os, subprocess
import sys
import termios

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

from PyQt5.QtCore import QLibraryInfo, QTimer, QSocketNotifier
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets


import signal

from dataclasses import dataclass
from collections import defaultdict

from typing import List, Dict, Any

from init.ann_index import AnnIndex



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

    def __init__(self, ip, frame_num, n, system, config_data):

        self.system = system

        self.camera = VideoCapture(ip, config_data=config_data)

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

        self.last_visitor_face_img = None

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

            original_frame = self.camera.read()

            if original_frame is None or original_frame.size == 0:

                time.sleep(0.01)

                continue

            

            # Resize the frame immediately to a consistent processing size

            resized_frame = cv2.resize(original_frame, (800, 600))

            

            # Store resized frame for detection and display, and high-res frame for recognition

            self.system.state.frame[self.frame_num] = resized_frame
            self.system.state.frame_high_res[self.frame_num] = original_frame

            now_frame = resized_frame.copy()

            font_size = 60



            # 繪製人臉框

            if self.system.state.max_box[self.frame_num] is not None:

                x1, y1, x2, y2 = self.system.state.max_box[self.frame_num]

                cv2.rectangle(now_frame, (x1, y1), (x2, y2), (255, 0, 0), 6)



                # 根據 same_class 狀態決定顯示文字

                current_class = self.system.state.same_class[self.frame_num]
                hint_msg = self.system.state.hint_text[self.frame_num]

                if hint_msg:
                    now_frame = put_chinese_text(now_frame, hint_msg, (x1, y1-55), font_path, font_size, (255, 85, 0)) # Orange for hint

                elif current_class == "__VISITOR__":

                    now_frame = put_chinese_text(now_frame, "訪客", (x1, y1-55), font_path, font_size, (0, 0, 255)) # Blue for visitor
                    
                    # Capture visitor face from high-resolution frame
                    try:
                        h_orig, w_orig, _ = original_frame.shape
                        h_small, w_small, _ = resized_frame.shape
                        scale_x = w_small / w_orig
                        scale_y = h_small / h_orig
                        
                        # Scale coordinates back to original resolution
                        orig_x1 = int(x1 / scale_x)
                        orig_x2 = int(x2 / scale_x)
                        orig_y1 = int(y1 / scale_y)
                        orig_y2 = int(y2 / scale_y)
                        
                        # Ensure coordinates are within bounds
                        fy1, fy2 = max(0, orig_y1), min(h_orig, orig_y2)
                        fx1, fx2 = max(0, orig_x1), min(w_orig, orig_x2)
                        
                        if fy2 > fy1 and fx2 > fx1:
                            self.last_visitor_face_img = original_frame[fy1:fy2, fx1:fx2].copy()
                    except Exception:
                        pass

                elif current_class != "None":

                    staff_name = self.system.state.features_dict.get("id_name", {}).get(current_class, "辨識中")

                    now_frame = put_chinese_text(now_frame, staff_name, (x1, y1-55), font_path, font_size, (205, 0, 0)) # Red for staff

                else:

                    now_frame = put_chinese_text(now_frame, "辨識中", (x1, y1-55), font_path, font_size, (0, 0, 0)) # Black for identifying



                # 觸發簽到/簽離 (API 呼叫)

                if self.system.state.same_people[self.frame_num] > 0:

                    LOGGER.info(f"成功辨識到人員，觸發後續的打卡流程。")

                    confidence = self.system.state.same_people[self.frame_num]

                    # 確保 current_class 是有效的員工ID，而不是 __VISITOR__

                    if current_class != "None" and current_class != "__VISITOR__":

                        success_staff_name = self.system.state.features_dict.get("id_name", {}).get(current_class, "未知員工")



                        if (not CONFIG["Clothes_detection"] or (self.system.state.clothes[0] and self.system.state.clothes[2])):

                            LOGGER.info(f"衣物偵測通過 (或未啟用)，準備呼叫 check_in_out。")

                            check_in_out(self.system, success_staff_name, current_class, self.frame_num, self.n_camera, confidence)

                        else:

                            LOGGER.warning(f"人員: {success_staff_name} 辨識成功 (信賴度: {confidence:.2%}), 但因衣物未穿戴整齊而跳過打卡。")

                        

                        self.save_img(self.system.state.frame[self.frame_num], "face", success_staff_name)



                    # 重設觸發器，防止重複呼叫

                    self.system.state.same_people[self.frame_num] = 0.0



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

            time.sleep(0.065)

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

        current_class = self.system.state.same_class[self.frame_num]



        if current_class == "__VISITOR__":

            # Prioritize showing the captured visitor face
            if self.last_visitor_face_img is not None and self.last_visitor_face_img.size > 0:
                try:
                    # Convert BGR (OpenCV) to RGB (Qt)
                    img_rgb = cv2.cvtColor(self.last_visitor_face_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    return QPixmap.fromImage(qimg)
                except Exception:
                    pass # Fallback to mask if conversion fails
            
            path = f'{main_path}/other/mask.png' # 使用通用圖示代表訪客

        elif current_class != "None":

            path = self.system.state.profile_dict.get(current_class, path)

        

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



        if current_class == "__VISITOR__":

            color = 'color: rgb(0, 0, 255); background-color: rgb(255, 255, 255); font: 24pt "微軟正黑體";'

            txt = "訪客"

        elif current_class != "None":

            color = 'color: rgb(0, 170, 0); background-color: rgb(255, 255, 255); font: 24pt "微軟正黑體";'

            txt = self.system.state.features_dict.get("id_name", {}).get(current_class, "辨識中")

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



    def terminate(self, event):

        print(f"Terminating CameraSystem for window {self.frame_num}...")

        self.stop_threads = True

        if hasattr(self, 'win'):

            self.win.my_thread.exit()

        self.camera.terminate()

        self.detect.terminate()

        self.compar.terminate()

        print(f"CameraSystem {self.frame_num} terminated. Accepting close event.")

        event.accept()



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

    same_people: List[float] = None

    same_class: List[str] = None

    frame: List[Any] = None

    frame_mtcnn: List[Any] = None
    
    frame_high_res: List[Any] = None # 儲存原始高解析度影像
    
    frame_mtcnn_high_res: List[Any] = None # 儲存與偵測同步的高解析度影像快照

    clothes: List[bool] = None

    check_time: Dict[str, List[Any]] = None

    features_dict: Dict[str, Any] = None

    profile_dict: Dict[str, str] = None

    display_history: List[list] = None # 用於在畫面上顯示歷史紀錄

    leave: int = 0

    min_face: List[int] = None

    max_points: List[Any] = None

    last_speak_time: Dict[str, float] = None

    ann_index: Any = None

    detection_interval: float = 0.1
    comparison_interval: float = 0.1
    
    hint_text: List[str] = None # 用於顯示即時提示 (例如: 請靠近鏡頭)



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

        self.state.same_people = [0.0, 0.0]

        self.state.same_class = ["None", "None"]

        self.state.frame = [None, None]

        self.state.frame_mtcnn = [None, None]
        
        self.state.frame_high_res = [None, None]
        
        self.state.frame_mtcnn_high_res = [None, None]

        self.state.clothes = [False, False, False]

        self.state.check_time = {}

        self.state.features_dict = {}

        self.state.profile_dict = {}

        self.state.display_history = [[], []]

        # 分別讀取入出口的最小人臉設定，若無則使用全域設定
        global_min_face = CONFIG.get("min_face", 100)
        in_min = CONFIG.get("inCamera", {}).get("min_face", global_min_face)
        out_min = CONFIG.get("outCamera", {}).get("min_face", global_min_face)
        self.state.min_face = [in_min, out_min]
        
        LOGGER.info(f"人臉最小像素設定 (Min Face): 全域={global_min_face}, 入口(Cam0)={in_min}, 出口(Cam1)={out_min}")

        self.state.max_points = [None, None]

        self.state.last_speak_time = {}

        self.state.ann_index = AnnIndex() # Initialize AnnIndex
        
        self.state.hint_text = ["", ""]



        self.mtcnn = mtcnn.MTCNN(image_size=160, min_face_size=95, keep_all=True, select_largest=True)

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




        # 確保所有必要的 media 子目錄都存在

        os.makedirs(os.path.join(self.local_media_path, "descriptors"), exist_ok=True)

        os.makedirs(os.path.join(self.local_media_path, "pic_bak"), exist_ok=True)

        os.makedirs(os.path.join(self.local_media_path, "profile_pictures"), exist_ok=True)

        self.update_lock = threading.Lock() # 用於防止多個更新執行緒同時運行

        # 執行效能自動調校
        self._auto_tune_performance()

    def _auto_tune_performance(self):
        """
        啟動時執行效能基準測試，根據 CPU 能力自動決定偵測與辨識的 FPS。
        """
        LOGGER.info("正在執行硬體效能自動調校 (Auto-Tuning)...")
        print("正在執行硬體效能自動調校...")
        
        # 1. 準備測試數據 (模擬真實運作情境)
        # 偵測用: 800x600 (因為主程式會先 resize 才丟進去)
        dummy_det_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        # 辨識用: 160x160 (ResNet 的標準輸入尺寸)
        dummy_rec_tensor = torch.randn(1, 3, 160, 160)

        # 2. 模型暖機
        try:
            self.mtcnn.detect(dummy_det_img)
            self.resnet(dummy_rec_tensor)
        except Exception:
            pass

        # 3. 測試偵測速度 (MTCNN @ 800x600)
        t0 = time.time()
        for _ in range(5):
            self.mtcnn.detect(dummy_det_img)
        avg_det_time = (time.time() - t0) / 5

        # 4. 測試辨識速度 (ResNet @ 160x160)
        t0 = time.time()
        for _ in range(5):
            self.resnet(dummy_rec_tensor)
        avg_rec_time = (time.time() - t0) / 5

        # 5. 計算建議間隔 (加入安全係數以保留 CPU 給 UI)
        # 偵測係數 3.0: 考慮雙鏡頭 + UI 開銷 + 系統抖動
        det_interval_raw = avg_det_time * 3.0
        # 辨識係數 1.5: 辨識較快，且非每幀觸發，係數可稍低
        rec_interval_raw = avg_rec_time * 1.5

        # 6. 設定上下限 (Max 15 FPS, Min 2 FPS)
        self.state.detection_interval = max(0.066, min(0.5, det_interval_raw))
        self.state.comparison_interval = max(0.066, min(0.5, rec_interval_raw))

        # 7. 輸出詳細日誌
        det_fps = 1 / self.state.detection_interval
        rec_fps = 1 / self.state.comparison_interval
        
        log_msg = (
            f"\n===== 效能調校結果 (Auto-Tuning) =====\n"
            f"CPU 單次偵測耗時 (800x600): {avg_det_time*1000:.2f} ms\n"
            f"CPU 單次辨識耗時 (160x160): {avg_rec_time*1000:.2f} ms\n"
            f"--------------------------------------\n"
            f"設定偵測頻率: {det_fps:.1f} FPS (間隔 {self.state.detection_interval:.3f}s)\n"
            f"設定辨識頻率: {rec_fps:.1f} FPS (間隔 {self.state.comparison_interval:.3f}s)\n"
            f"======================================"
        )
        LOGGER.info(log_msg)
        print(log_msg)

    def run(self):
        """
        啟動系統主流程：
        - 建立 PyQt Application。
        - 設定安全的訊號處理機制。
        - 載入模型、啟動背景更新、建立攝影機。
        - 進入 PyQt 事件主迴圈。
        """
        app = QApplication(sys.argv)

        # Set up a pipe-based mechanism for safe shutdown from signals.
        # This is the most robust way to handle POSIX signals in a Qt app.
        safe_shutdown_pipe_read, self.safe_shutdown_pipe_write = os.pipe()

        def signal_handler(sig, frame):
            print(f"接收到訊號 {signal.Signals(sig).name}，觸發安全關閉...")
            # Write a byte to the pipe. This is an async-signal-safe operation.
            os.write(self.safe_shutdown_pipe_write, b'x')

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # QSocketNotifier will watch the read end of the pipe in the Qt event loop.
        self.shutdown_notifier = QSocketNotifier(safe_shutdown_pipe_read, QSocketNotifier.Read)
        # We need to manually add self as a context for lambda to have access to it, even if it's not a QObject
        self.shutdown_notifier.activated.connect(lambda: (os.read(safe_shutdown_pipe_read, 1), self._safe_shutdown()))




        print("系統啟動中...")

        # 1. 嘗試載入本地現有模型



        self._load_features_and_profiles()



        # 2. 在背景啟動資料更新流程

        print("在背景啟動首次資料同步...")

        threading.Thread(target=self.update_data_and_model, daemon=True).start()



        # 3. 啟動UI和攝影機

        self.update_inout_log()

        self.setup_cameras()

        

        # 4. 進入 PyQt 事件主迴圈
        try:
            ret = app.exec_()
        finally:
            # 使用延遲的子程序來修復終端機，避開 PyQt 清理過程的覆蓋
            subprocess.Popen("sleep 0.1; stty sane", shell=True)
            try:
                sys.exit(ret)
            except UnboundLocalError:
                sys.exit(0)

    def update_data_and_model(self, initial_run=True):

        """

        資料更新與模型訓練的主流程，設計為可在背景執行緒中安全運行。

        """

        if not self.update_lock.acquire(blocking=False):

            print("更新流程已在運行中，跳過本次觸發。")

            return



        try:

            print("\n===== 開始背景資料更新流程 ====")

            # 步驟 1: 與遠端伺服器同步檔案

            sync_success = self._sync_files_with_server()

            if not sync_success:

                print("檔案同步失敗，中止本次更新。")

                return



            pic_bak_path = os.path.join(self.local_media_path, "pic_bak")

            descriptors_path = os.path.join(self.local_media_path, "descriptors")



            # 步驟 2: 找出新增和刪除的圖片

            pic_map = {os.path.splitext(f)[0]: f for f in os.listdir(pic_bak_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

            local_pics_basenames = set(pic_map.keys())

            local_descriptors_basenames = set(os.path.splitext(f)[0] for f in os.listdir(descriptors_path) if f.lower().endswith('.npy'))



            new_basenames = local_pics_basenames - local_descriptors_basenames

            deleted_basenames = local_descriptors_basenames - local_pics_basenames



            new_files = [pic_map[name] for name in new_basenames]

            deleted_files = [f"{name}.npy" for name in deleted_basenames]



            update_needed = bool(new_files or deleted_files)

            print(f"檔案比對完成：發現 {len(new_files)} 張新圖片，{len(deleted_files)} 個過期特徵檔。")



            # 步驟 3: 處理檔案變動

            if update_needed:

                self._process_deleted_descriptors(deleted_files, descriptors_path)

                self._generate_new_descriptors(new_files, pic_bak_path, descriptors_path)



                # 步驟 4: 重新載入特徵並訓練模型

                features, X_train, y_train, X_test, y_test = self._load_features_from_disk()




                # 步驟 5: 熱更新系統狀態

                self.state.features_dict = features

                self._update_profile_pictures()

                self._load_or_build_index(force_rebuild=True) # Force rebuild Faiss index

                print("模型與資料已熱更新完畢。")

            else:

                print("資料無變動，無需更新模型。")



            # 無論資料是否變動，都執行一次數量更新，以確保與伺服器的資料一致性

            # 改為非同步執行，避免阻塞更新主流程

            threading.Thread(target=self._update_descriptor_counts_on_server, daemon=True).start()



        finally:

            self.update_lock.release()

            # 如果不是初次運行，則設定下一次的定時更新

            if not initial_run:

                print(f"===== 背景資料更新流程結束，將在 300 秒後再次執行 ====")



            # 設定下一次定時器

            timer = threading.Timer(300, self.update_data_and_model, args=[False])

            timer.daemon = True

            timer.start()


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

                

                boxes, probs, points = self.mtcnn.detect(image, landmarks=True)

                

                if boxes is not None and points is not None:

                    box = boxes[0]

                    point = points[0]

                    

                    img_cropped = crop_face_without_forehead(image, box, point)

                    

                    img_embedding = self.resnet(img_cropped.unsqueeze(0))

                    

                    basename = os.path.splitext(filename)[0]

                    np.save(os.path.join(descriptors_path, f"{basename}.npy"), img_embedding[0].detach().numpy())

                else:

                    LOGGER.warning(f"在圖片 {filename} 中未偵測到人臉或特徵點，跳過此檔案。")



            except Exception as e:

                LOGGER.error(f"處理圖片 {filename} 時發生錯誤: {e}")


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





    def _load_features_and_profiles(self):

        """載入本地現有的特徵和頭像資料"""

        try:

            features, _, _, _, _ = self._load_features_from_disk()

            self.state.features_dict = features

            self._update_profile_pictures()

            self._load_or_build_index(force_rebuild=False) # Attempt to load or build index

            print("本地特徵與頭像資料載入完成。")

        except Exception as e:

            print(f"載入本地特徵時發生錯誤: {e}")


    def _update_profile_pictures(self):

        """更新大頭照字典，改為從 pic_bak 中根據檔名日期尋找每個人的最新照片"""

        pic_bak_path = os.path.join(self.local_media_path, "pic_bak")

        if not os.path.isdir(pic_bak_path):

            self.state.profile_dict = {}

            return



        latest_pics = {}

        for filename in os.listdir(pic_bak_path):

            try:

                base_name = os.path.splitext(filename)[0]

                parts = base_name.split('_', 1)

                if len(parts) < 2:

                    continue # 忽略不含底線的檔名



                category = parts[0].upper() # 強制轉為大寫以確保一致性

                sort_key = parts[1] # e.g., "2025_1114_0319_50_421000_Yang"



                # 因為時間戳是左邊對齊且格式固定，可以直接用字串比較來找到最新的

                if category not in latest_pics or sort_key > latest_pics[category]['sort_key']:

                    latest_pics[category] = {

                        'path': os.path.join(pic_bak_path, filename),

                        'sort_key': sort_key

                    }

            except IndexError:

                # 忽略格式不符的檔案

                continue

        

        # 建立最終的 profile_dict

        self.state.profile_dict = {cat: data['path'] for cat, data in latest_pics.items()}


    # setup_cameras, update_inout_log, shutdown 等方法維持不變

    def _load_or_build_index(self, force_rebuild=False):

        """

        Loads the Faiss index if it exists and is consistent, otherwise builds a new one.

        """

        if not self.state.features_dict or not any(self.state.features_dict.values()):

            print("沒有人臉特徵資料可供建立 Faiss 索引，跳過。")

            return



        print("正在準備 Faiss 索引...")

        if not force_rebuild and self.state.ann_index.load():

            # Check if the number of vectors in index matches the features_dict

            # Sum of lengths of all embedding lists, excluding 'id_name'

            current_features_count = sum(len(v) for k, v in self.state.features_dict.items() if k != 'id_name')

            if self.state.ann_index.index is not None and self.state.ann_index.index.ntotal == current_features_count:

                print("已成功載入現有的 Faiss 索引。")

            else:

                print("Faiss 索引與特徵數量不匹配或索引檔案損壞，強制重建...")

                self.state.ann_index.build(self.state.features_dict)

        else:

            if force_rebuild:

                print("強制重建 Faiss 索引...")

            else:

                print("找不到現有的 Faiss 索引，將建立新的索引。")

            self.state.ann_index.build(self.state.features_dict)


    def _get_valid_staff_ids(self):

        """從 API 獲取合法的員工 ID 總名單。"""

        api_url = CONFIG["Server"]["API_url"]

        staffs_endpoint = f"{api_url}/staffs/"

        try:

            response = requests.get(staffs_endpoint, timeout=10)

            response.raise_for_status()  # Will raise an exception for 4xx/5xx status

            staff_data = response.json()

            valid_ids = {item["staff_id"] for item in staff_data}

            LOGGER.info(f"從伺服器成功獲取 {len(valid_ids)} 位合法員工ID。")

            return valid_ids

        except requests.exceptions.RequestException as e:

            LOGGER.error(f"從 {staffs_endpoint} 獲取員工總名單失敗: {e}")

            return None

        except Exception as e:

            LOGGER.error(f"解析員工總名單時發生錯誤: {e}")

            return None


    def _update_descriptor_counts_on_server(self):

        """

        計算每位人員的 .npy 檔案數量，並只對合法員工透過 PATCH 請求更新到伺服器。

        """

        # 步驟 1: 從伺服器獲取合法的員工ID列表

        valid_staff_ids = self._get_valid_staff_ids()

        if valid_staff_ids is None:

            print("無法獲取合法員工名單，中止特徵檔數量更新。")

            LOGGER.error("無法獲取合法員工名單，中止特徵檔數量更新。")

            return



        print("開始更新伺服器上的人員特徵檔數量(僅限合法員工)...")

        LOGGER.info("開始更新伺服器上的人員特徵檔數量(僅限合法員工)...")



        descriptors_path = os.path.join(self.local_media_path, "descriptors")

        if not os.path.isdir(descriptors_path):

            LOGGER.warning("特徵檔目錄不存在，跳過更新。")

            return



        # 步驟 2: 計算本地每個 staff_id 的檔案數量

        staff_counts = defaultdict(int)

        for filename in os.listdir(descriptors_path):

            if filename.lower().endswith('.npy'):

                try:

                    staff_id = filename.split('_')[0]

                    staff_counts[staff_id] += 1

                except IndexError:

                    continue



        # 步驟 3: 只遍歷合法的員工ID，並更新他們的計數

        api_url = CONFIG["Server"]["API_url"]

        updated_count = 0

        failed_staff = []



        for staff_id in valid_staff_ids:

            count = staff_counts.get(staff_id, 0)  # 如果本地沒有檔案，則數量為0

            url = f"{api_url}/staffs/{staff_id}/"

            payload = {"descriptors": count}

            

            try:

                response = requests.patch(url, json=payload, timeout=5)

                if response.status_code == 200:

                    # 成功的日誌可以選擇性記錄，避免過多訊息

                    # LOGGER.info(f"成功更新人員 {staff_id} 的特徵檔數量為 {count}。")

                    updated_count += 1

                else:

                    LOGGER.error(f"更新人員 {staff_id} 失敗。伺服器回應: {response.status_code} - {response.text}")

                    failed_staff.append(staff_id)

            except requests.exceptions.RequestException as e:

                LOGGER.error(f"向 {url} 發送 PATCH 請求時發生網路錯誤: {e}")

                failed_staff.append(staff_id)



        print(f"特徵檔數量更新完成。在 {len(valid_staff_ids)} 位合法員工中，成功更新 {updated_count} 位。")

        LOGGER.info(f"特徵檔數量更新完成。在 {len(valid_staff_ids)} 位合法員工中，成功更新 {updated_count} 位。")

        if failed_staff:

            LOGGER.warning(f"更新失敗的合法員工ID: {failed_staff}")

            print(f"更新失敗的合法員工ID: {failed_staff}")


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

                self.cameras.append(CameraSystem(ip, i, n, self, CONFIG))



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

                    leave_timer = threading.Timer(5,clear_leave_employee, (self, staff_id, ))

                    leave_timer.daemon = True

                    leave_timer.start()

            print(date_str, "更新已簽到名單")

            LOGGER.info("更新已簽到名單")



        except Exception as e:

            LOGGER.info("已簽到名單更新失敗"+str(e))

            print(date_str, "已簽到名單更新失敗", e)

            updata_log_timer = threading.Timer(300, self.update_inout_log)

            updata_log_timer.daemon = True

            updata_log_timer.start()


    def _safe_shutdown(self):
        """
        在 Qt 事件迴圈中安全地執行關閉程序。
        """
        print("\nShutting down... Closing all windows.")
        # Closing all windows will trigger their respective closeEvents,
        # which are connected to the `terminate` method of each CameraSystem.
        QApplication.closeAllWindows()

if __name__ == "__main__":

    try:

        face_recognition_system = FaceRecognitionSystem()

        face_recognition_system.run()

    except Exception as e:

        print(f"An unexpected error occurred during startup: {e}")

        # In case of startup error before app.exec_(), just exit.

        os._exit(1)