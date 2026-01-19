from pathlib import Path
import time, threading, queue, json, os, subprocess, shutil
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor # [2026-01-13 Perf]
import paho.mqtt.client as mqtt
import sys, termios, warnings, signal
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
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
import datetime
from init.camera import VideoCapture
from init.model import Detector, Comparison
from function import *
from PyQt5.QtCore import QLibraryInfo, QTimer, QSocketNotifier
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any
from init.ann_index import AnnIndex

main_path = os.path.dirname(__file__)
def check_empty_string_in_dict(data):
    for key, value in data.items():
        if isinstance(value, dict):
            if not check_empty_string_in_dict(value): return False
        elif value == "": return False
    return True

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
font_path = os.path.join(os.path.dirname(__file__), "other/NotoSansTC-VariableFont_wght.ttf")
CAMERA = {0:"inCamera", 1:"outCamera"}

def put_chinese_text(img, text, position, font_path, font_size, color, background=True):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox(position, text, font=font)
    if background: draw.rectangle(text_bbox, fill='white')
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

class CameraSystem:
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
        if frame_num == 1 and n < 2: return
        self.win = MainWindow(self.updata_screen, frame_num)
        self.img1_size = (self.win.img1.width(), self.win.img1.height())
        self.img2_size = (self.win.img2.width(), self.win.img2.height())
        if n == 1: self.win.setWindowTitle(f"進出視窗")
        self.win.closeEvent = self.terminate
        if self.frame_num == 0 and CONFIG["Clothes_show"]:
            self.win.img3.setPixmap(QPixmap(f'{main_path}/other/helmet_R.png'))
            self.win.img4.setPixmap(QPixmap(f'{main_path}/other/vest_R.png'))
            self.win.img3.setStyleSheet("QLabel{background-color: rgba(255,0,0,255);}")
            self.win.img4.setStyleSheet("QLabel{background-color: rgba(255,0,0,255);}")
            self.win.img3.setScaledContents(True)
            self.win.img4.setScaledContents(True)

    def main_camera(self):
        frame_count = 0
        while not self.stop_threads:
            original_frame = self.camera.read()
            if original_frame is None or original_frame.size == 0:
                time.sleep(0.01); continue
            
            # 更新 AI 與存檔用的原圖
            self.system.state.frame[self.frame_num] = original_frame
            self.system.state.frame_high_res[self.frame_num] = original_frame
            
            # [效能優化] 先縮圖，再繪圖 (Process Small, Display Small, Save Big)
            # 原本在 1080p 上做 PIL 中文繪圖太慢，導致延遲。
            # 改為縮小到 960px 寬 (1/2 尺寸, 1/4 像素量)，速度提升 4 倍。
            h, w = original_frame.shape[:2]
            target_w = 960
            scale = target_w / w if w > target_w else 1.0
            
            if scale < 1.0:
                target_h = int(h * scale)
                now_frame = cv2.resize(original_frame, (target_w, target_h))
            else:
                now_frame = original_frame.copy()

            font_size = int(60 * scale) # 字體也跟著縮放
            if font_size < 20: font_size = 20

            if self.system.state.max_box[self.frame_num] is not None:
                # 座標轉換: 原圖 -> 小圖
                ox1, oy1, ox2, oy2 = self.system.state.max_box[self.frame_num]
                x1, y1, x2, y2 = int(ox1*scale), int(oy1*scale), int(ox2*scale), int(oy2*scale)
                
                cv2.rectangle(now_frame, (x1, y1), (x2, y2), (255, 0, 0), max(2, int(6*scale)))
                current_class = self.system.state.same_class[self.frame_num]
                hint_msg = self.system.state.hint_text[self.frame_num]
                text_y = y1 - int(55*scale) if y1 - int(55*scale) > 10 else y2 + 10
                
                if hint_msg:
                    now_frame = put_chinese_text(now_frame, hint_msg, (x1, text_y), font_path, font_size, (255, 85, 0))
                elif current_class == "__VISITOR__":
                    now_frame = put_chinese_text(now_frame, "訪客", (x1, text_y), font_path, font_size, (0, 0, 255))
                    try:
                        # 訪客頭像截取仍需使用原圖 (保持解析度)
                        if oy2 > oy1 and ox2 > ox1: 
                             self.last_visitor_face_img = original_frame[max(0,oy1):min(h,oy2), max(0,ox1):min(w,ox2)].copy()
                    except Exception: pass
                elif current_class != "None":
                    self.last_visitor_face_img = None # [2026-01-19 Fix] Reset visitor img to avoid showing previous person
                    staff_name = self.system.state.features_dict.get("id_name", {}).get(current_class, "辨識中")
                    now_frame = put_chinese_text(now_frame, staff_name, (x1, text_y), font_path, font_size, (205, 0, 0))
                else:
                    self.last_visitor_face_img = None # [2026-01-19 Fix] Reset visitor img
                    now_frame = put_chinese_text(now_frame, "辨識中", (x1, text_y), font_path, font_size, (0, 0, 0))
                
                # 辨識後處理邏輯 (保持不變)
                if self.system.state.same_people[self.frame_num] > 0:
                    confidence = self.system.state.same_people[self.frame_num]
                    if current_class not in ["None", "__VISITOR__"]:
                        success_staff_name = self.system.state.features_dict.get("id_name", {}).get(current_class, "未知員工")
                        if (not CONFIG["Clothes_detection"] or (self.system.state.clothes[0] and self.system.state.clothes[2])):
                            check_in_out(self.system, success_staff_name, current_class, self.frame_num, self.n_camera, confidence)
                        z_score = self.system.state.same_zscore[self.frame_num]
                        width_val = self.system.state.same_width[self.frame_num]
                        saved_img = self.system.state.success_frame[self.frame_num]
                        if saved_img is not None: self.save_img(saved_img, "face", success_staff_name, confidence, z_score, width_val)
                        else: self.save_img(self.system.state.frame_high_res[self.frame_num], "face", success_staff_name, confidence, z_score, width_val)
                    self.system.state.same_people[self.frame_num] = 0.0
            
            self.show_frame = now_frame

    def updata_screen(self):
        time.sleep(0.5)
        self.win.my_thread.signal_update_img.connect(self.win.update_img)
        self.win.my_thread.signal_update_hint.connect(self.win.update_hint)
        if self.clothes_de: self.win.my_thread.signal_update_bgcolor.connect(self.win.update_bgcolor)
        while not self.stop_threads:
            if self.show_frame.shape[0] == 0: continue
            try:
                self.win.my_thread.signal_update_img.emit(self.win.img1, self.show_main())
                self.win.my_thread.signal_update_img.emit(self.win.img2, self.shwo_head())
                if self.clothes_de:
                    img3, img4 = self.show_save()
                    self.win.my_thread.signal_update_img.emit(self.win.img3, img3); self.win.my_thread.signal_update_img.emit(self.win.img4, img4)
                color, txt = self.show_hint()
                self.win.my_thread.signal_update_hint.emit(self.win.hint, color, txt)
            except Exception: pass
            time.sleep(0.065)

    def show_main(self):
        alpha = 0.1
        show_img = cv2.addWeighted(cv2.resize(self.image, (self.show_frame.shape[1], self.show_frame.shape[0])), alpha, self.show_frame, 1 - alpha, 0)
        show_img = cv2.resize(show_img, self.img1_size)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        h, w, ch = show_img.shape
        return QPixmap.fromImage(QImage(show_img, w, h, ch * w, QImage.Format_RGB888))

    def shwo_head(self):
        path = f'{main_path}/other/clear_img.png'
        current_class = self.system.state.same_class[self.frame_num]
        if current_class == "__VISITOR__":
            if self.last_visitor_face_img is not None and self.last_visitor_face_img.size > 0:
                try:
                    img_rgb = cv2.cvtColor(self.last_visitor_face_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    return QPixmap.fromImage(QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888))
                except Exception: pass
            path = f'{main_path}/other/mask.png'
        elif current_class != "None":
            path = self.system.state.profile_dict.get(current_class, path)
        return QPixmap(path)

    def show_save(self):
        h, v = "helmet_R.png", "vest_R.png"
        if self.system.state.clothes[0]: v = "vest_G.png"
        if self.system.state.clothes[2]: h = "helmet_G.png"
        return QPixmap(f'{main_path}/other/{h}'), QPixmap(f'{main_path}/other/{v}')

    def show_hint(self):
        current_class = self.system.state.same_class[self.frame_num]
        if current_class == "__VISITOR__":
            return 'color: rgb(0, 0, 255); background-color: rgb(255, 255, 255); font: 24pt "微軟正黑體";', "訪客"
        elif current_class != "None":
            name = self.system.state.features_dict.get("id_name", {}).get(current_class, "辨識中")
            return 'color: rgb(0, 170, 0); background-color: rgb(255, 255, 255); font: 24pt "微軟正黑體";', name
        return 'color: rgb(0, 85, 255); background-color: rgb(255, 255, 255); font: 24pt "微軟正黑體";', "辨識中"

    def save_img(self, img, path, staffname="", conf=0.0, z_score=0.0, width=0):
        # [2026-01-13 Perf] Offload disk I/O to background thread
        self.system.io_pool.submit(self._save_img_task, img.copy(), path, staffname, conf, z_score, width)

    def _save_img_task(self, img, path, staffname, conf, z_score, width):
        try:
            dict_={"face":0, "clothes":1}
            dt = datetime.datetime.today()
            d_str, t_str = dt.strftime("%Y_%m_%d"), dt.strftime("%H;%M;%S")
            os.makedirs(f"{main_path}/img_log/{path}/{d_str}", exist_ok=True)
            if time.time()-self.save_img_time[dict_[path]] > 5 or (self.save_name_last != staffname and staffname != ""):
                fname = f"{t_str}_{staffname}_C{int(conf*100)}_Z{z_score:.2f}_W{width}.jpg" if staffname else f"{t_str}.jpg"
                cv2.imwrite(f"{main_path}/img_log/{path}/{d_str}/{fname}", img)
                self.save_img_time[dict_[path]] = time.time()
                if staffname: self.save_name_last = staffname
        except Exception as e:
            LOGGER.error(f"Async save_img failed: {e}")

    def terminate(self, event):
        print(f"Terminating CameraSystem for window {self.frame_num}...")
        self.stop_threads = True
        
        # [2026-01-19 Fix] Wait for UI thread worker
        if hasattr(self, 'win') and hasattr(self.win, 'my_thread'): 
             self.win.my_thread.exit()
             self.win.my_thread.wait(100) # Wait max 100ms

        # Terminate components
        self.camera.terminate()
        self.detect.terminate() 
        self.compar.terminate() 
        
        event.accept()

with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)
if not check_empty_string_in_dict(CONFIG): os._exit(0)
API = config.API(str(CONFIG["Server"]["API_url"]), int(CONFIG["Server"]["location_ID"]))

@dataclass
class GlobalState:
    max_box: List[Any] = None
    same_people: List[float] = None
    same_zscore: List[float] = None
    same_width: List[int] = None
    same_class: List[str] = None
    frame: List[Any] = None
    frame_mtcnn: List[Any] = None
    frame_high_res: List[Any] = None
    frame_mtcnn_high_res: List[Any] = None
    success_frame: List[Any] = None
    clothes: List[bool] = None
    check_time: Dict[str, List[Any]] = None
    features_dict: Dict[str, Any] = None
    profile_dict: Dict[str, str] = None
    display_history: List[list] = None
    leave: int = 0
    min_face: List[int] = None
    max_points: List[Any] = None
    last_speak_time: Dict[str, float] = None
    ann_index: Any = None
    detection_interval: float = 0.1
    comparison_interval: float = 0.1
    hint_text: List[str] = None
    gaze_status: List[Any] = None
    frame_data: List[Any] = None
    head_pose: List[Any] = None
    part_features: Dict[str, Dict[str, Any]] = None # [2026-01-19] Part-based features for verification

class FaceRecognitionSystem:
    def __init__(self):
        self.n_camera = 2
        self.state = GlobalState()
        self.state.max_box = [None] * self.n_camera
        self.state.same_people = [0.0] * self.n_camera
        self.state.same_zscore = [0.0] * self.n_camera
        self.state.same_width = [0] * self.n_camera
        self.state.same_class = ["None"] * self.n_camera
        self.state.frame, self.state.frame_mtcnn = [None, None], [None, None]
        self.state.frame_high_res, self.state.frame_mtcnn_high_res = [None, None], [None, None]
        self.state.success_frame = [None, None]
        self.state.clothes = [False, False, False]
        self.state.check_time, self.state.features_dict, self.state.profile_dict = {}, {}, {}
        self.state.part_features = {} # Initialize empty
        self.state.display_history = [[], []]
        gm = CONFIG.get("min_face", 100)
        self.state.min_face = [CONFIG.get("inCamera",{}).get("min_face", gm), CONFIG.get("outCamera",{}).get("min_face", gm)]
        self.state.max_points = [None, None]
        self.state.last_speak_time = {}
        self.state.ann_index = AnnIndex()
        self.state.hint_text = ["", ""]
        self.state.gaze_status = [None] * self.n_camera
        self.state.frame_data = [None] * self.n_camera
        self.state.head_pose = [None] * self.n_camera
        self.mp_detectors = {} 
        self.resnet = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2').eval()
        if CONFIG["Clothes_show"]:
            models_dir = Path(f'{os.path.dirname(__file__)}/models')
            model_name = 'best_cloth2'
            int8_model_det_path = models_dir/'int8'/f'{model_name}_openvino_model/{model_name}.xml'
            self.model_clothes = YOLOv10(int8_model_det_path.parent, task='detect')
        self.speaker = Say_()
        self.local_media_path = os.path.join(os.path.dirname(__file__), "media")
        for d in ["descriptors", "pic_bak", "profile_pictures"]: os.makedirs(os.path.join(self.local_media_path, d), exist_ok=True)
        self.update_lock = threading.Lock()
        
        # [2026-01-13 Perf] Thread pool for non-blocking I/O (e.g., image saving)
        # Prevents main loop from stalling during disk writes.
        self.io_pool = ThreadPoolExecutor(max_workers=2)
        
        # Blocking asset rebuild on startup (Voice -> Descriptors)
        self._rebuild_assets()
        
        # Load features AFTER rebuild is complete
        self._load_features_and_profiles()
        
        # [Perf] Enable auto-tune to maximize runtime FPS based on hardware capability
        self._auto_tune_performance()

    def _rebuild_assets(self):
        LOGGER.info("Starting mandatory asset rebuild...")
        self._sync_files_with_server()
        dp = os.path.join(self.local_media_path, "descriptors")
        pb = os.path.join(self.local_media_path, "pic_bak")
        vp = os.path.join(main_path, "voice")

        # 1. Voice (IO Bound) - Run FIRST
        if os.path.exists(vp): shutil.rmtree(vp)
        os.makedirs(vp, exist_ok=True)
        
        if os.path.isdir(pb):
            LOGGER.info("Stage 1/2: Rebuilding voice files...")
            generic_texts = {}
            for key, val in CONFIG.get("say", {}).items():
                txt = val.replace("name_", "") if "name_" in val else val
                generic_texts[key] = txt

            names = set()
            for f in os.listdir(pb):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try: names.add(os.path.splitext(f)[0].split("_")[-1])
                    except: pass

            # Collect all tasks: (filename, text)
            tasks = []
            # 1. Generic commands (e.g., _in.mp3)
            for key, txt in generic_texts.items():
                tasks.append((f"_{key}.mp3", txt))
            # 2. Person-specific greetings (e.g., Yang_in.mp3)
            for name in names:
                for key, txt in generic_texts.items():
                    tasks.append((f"{name}_{key}.mp3", f"{name}{txt}"))

            def gen_one_voice(filename, text):
                try:
                    tts = gTTS(text=text, lang='zh-tw')
                    tts.save(os.path.join(vp, filename))
                except Exception: pass

            # Use ThreadPoolExecutor for all tasks (Blocking wait)
            # [Perf] Maximize IO parallelism (default workers = CPU_count + 4)
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(gen_one_voice, fn, txt) for fn, txt in tasks]
                for f in tqdm(futures, desc="[Voice Gen     ]"): 
                    try: f.result()
                    except Exception: pass

        # 2. Descriptors (CPU Bound) - Run SECOND
        if os.path.exists(dp): shutil.rmtree(dp)
        os.makedirs(dp, exist_ok=True)
        
        if os.path.isdir(pb):
            pic_files = [f for f in os.listdir(pb) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if pic_files:
                LOGGER.info(f"Stage 2/2: Rebuilding descriptors for {len(pic_files)} images...")
                from models.mtcnn import MTCNN
                detector = MTCNN()
                for f in tqdm(pic_files, desc="[Descriptor Gen]"):
                    try:
                        img = Image.open(os.path.join(pb, f)).convert('RGB')
                        boxes, _, points = detector.detect(img, landmarks=True)
                        if boxes is not None:
                            emb = self.resnet(crop_face_without_forehead(img, boxes[0], points[0]).unsqueeze(0))
                            np.save(os.path.join(dp, f"{os.path.splitext(f)[0]}.npy"), emb[0].detach().numpy())
                    except Exception: pass
        
        LOGGER.info("Assets rebuild complete.")

    def _auto_tune_performance(self):
        img, ts = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8), torch.randn(1, 3, 160, 160)
        from init.mediapipe_handler import MediaPipeHandler
        mp = MediaPipeHandler()
        try: mp.detect(img); self.resnet(ts)
        except Exception: pass
        t0 = time.time()
        for _ in range(5): mp.detect(img)
        dt = (time.time() - t0) / 5
        t0 = time.time()
        for _ in range(5): self.resnet(ts)
        rt = (time.time() - t0) / 5
        self.state.detection_interval = max(0.066, min(0.5, dt * 2.0))
        self.state.comparison_interval = max(0.066, min(0.5, rt * 1.5))
        LOGGER.info(f"Auto-Tuning: Det {1/self.state.detection_interval:.1f} FPS, Rec {1/self.state.comparison_interval:.1f} FPS")

    def run(self):
        # [2026-01-19 Fix TTY] Backup terminal settings at startup
        try:
            self.original_tty_settings = termios.tcgetattr(sys.stdin)
        except Exception: 
            self.original_tty_settings = None

        app = QApplication(sys.argv)
        safe_shutdown_pipe_read, self.safe_shutdown_pipe_write = os.pipe()
        def signal_handler(sig, frame): os.write(self.safe_shutdown_pipe_write, b'x')
        signal.signal(signal.SIGINT, signal_handler); signal.signal(signal.SIGTERM, signal_handler)
        self.shutdown_notifier = QSocketNotifier(safe_shutdown_pipe_read, QSocketNotifier.Read)
        self.shutdown_notifier.activated.connect(lambda: (os.read(safe_shutdown_pipe_read, 1), self._safe_shutdown()))
        self._load_features_and_profiles(); self.setup_mqtt_client(); self.update_inout_log(); self.setup_cameras()
        try: ret = app.exec_()
        finally:
            # [2026-01-19 Fix] Force exit to prevent hanging/high-load due to daemon threads spinning
            # or C++ resource cleanup issues (OpenCV/MediaPipe).
            
            # [2026-01-19 Fix TTY] Synchronously restore terminal state BEFORE exit using Python termios
            # This is more reliable than os.system('stty sane')
            if self.original_tty_settings:
                print("Restoring terminal settings via termios...")
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_tty_settings)
                except Exception: pass
            
            # [2026-01-19 Fix TTY] Launch a detached "rescuer" process.
            # Even if we restore termios above, background C++ threads (OpenCV/FFmpeg)
            # might corrupt the TTY during the final os._exit().
            # This external 'stty sane' will run 0.1s AFTER we die to clean up any mess.
            # start_new_session=True ensures it survives our os._exit().
            try:
                subprocess.Popen(
                    "sleep 0.1; stty sane", 
                    shell=True, 
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception: pass
            
            print("Force exiting system...")
            os._exit(0)

    def setup_mqtt_client(self):
        s_ip = CONFIG.get("Server", {}).get("ip", "localhost")
        m_conf = CONFIG.get("MQTT", {})
        self.mqtt_broker_host = m_conf.get("broker_ip", s_ip)
        self.mqtt_port, self.mqtt_topic = m_conf.get("port", 1883), m_conf.get("topic", "pvms/faces/updated")
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = self.on_mqtt_connect; client.on_message = self.on_mqtt_message
        try: client.connect(self.mqtt_broker_host, self.mqtt_port, 60); client.loop_start()
        except Exception: pass

    def on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0: client.subscribe(self.mqtt_topic); self.update_data_and_model(initial_run=True)

    def on_mqtt_message(self, client, userdata, msg):
        try: self.update_data_and_model(initial_run=False)
        except Exception: pass

    def update_data_and_model(self, initial_run=True):
        if not self.update_lock.acquire(blocking=False): return
        try:
            if not self._sync_files_with_server(): return
            pb, dp = os.path.join(self.local_media_path, "pic_bak"), os.path.join(self.local_media_path, "descriptors")
            pic_map = {os.path.splitext(f)[0]: f for f in os.listdir(pb) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
            local_desc = set(os.path.splitext(f)[0] for f in os.listdir(dp) if f.lower().endswith('.npy'))
            new_files = [pic_map[b] for b in pic_map if b not in local_desc or os.path.getmtime(os.path.join(pb, pic_map[b])) > os.path.getmtime(os.path.join(dp, f"{b}.npy"))]
            deleted = [f"{n}.npy" for n in local_desc - set(pic_map.keys())]
            if new_files or deleted:
                for f in deleted: 
                    try: os.remove(os.path.join(dp, f))
                    except Exception: pass
                self._generate_new_descriptors(new_files, pb, dp)
                f, _, _, _, _ = self._load_features_from_disk(); self.state.features_dict = f
                self._update_profile_pictures(); self._load_or_build_index(force_rebuild=True)
            threading.Thread(target=self._update_descriptor_counts_on_server, daemon=True).start()
        finally: self.update_lock.release()

    def _sync_files_with_server(self):
        s_dir = CONFIG["Server"]["face_data_dir"]
        s_conf = {"ip": CONFIG["Server"]["ip"], "username": CONFIG["Server"]["username"]}
        sp, dp = os.path.join(s_dir, "pic_bak").replace('\\', '/'), os.path.join(self.local_media_path, "pic_bak").replace('\\', '/')
        if not ssh.sync_with_rsync(s_conf, sp, dp): return False
        sp, dp = os.path.join(s_dir, "profile_pictures").replace('\\', '/'), os.path.join(self.local_media_path, "profile_pictures").replace('\\', '/')
        return ssh.sync_with_rsync(s_conf, sp, dp)

    def _process_deleted_descriptors(self, deleted_files, descriptors_path):
        for f in deleted_files:
            try: os.remove(os.path.join(descriptors_path, f))
            except OSError: pass

    def _generate_new_descriptors(self, new_files, pb, dp):
        from models.mtcnn import MTCNN
        detector = MTCNN()
        for f in tqdm(new_files, desc="[1/2] Descriptor Generation"):
            try:
                img = Image.open(os.path.join(pb, f)).convert('RGB')
                boxes, _, points = detector.detect(img, landmarks=True)
                if boxes is not None:
                    emb = self.resnet(crop_face_without_forehead(img, boxes[0], points[0]).unsqueeze(0))
                    np.save(os.path.join(dp, f"{os.path.splitext(f)[0]}.npy"), emb[0].detach().numpy())
            except Exception: pass

    def _load_features_from_disk(self):
        dp = os.path.join(self.local_media_path, "descriptors")
        xt, yt, xv, yv = [], [], [], []
        feat = {"id_name": {}}
        for f in [f for f in os.listdir(dp) if f.lower().endswith('.npy')]:
            cat, name = f.split("_")[0], f.split("_")[-1].split(".")[0]
            load = np.load(os.path.join(dp, f))
            if cat not in feat:
                feat[cat] = []; xv.append(load); yv.append(cat)
            else: xt.append(load); yt.append(cat)
            feat[cat].append(load); feat["id_name"][cat] = name
        return feat, xt, yt, xv, yv

    def _load_features_and_profiles(self):
        try:
            f, _, _, _, _ = self._load_features_from_disk()
            self.state.features_dict = f; self._update_profile_pictures(); self._load_or_build_index(force_rebuild=False)
            self._load_part_features() # [2026-01-19] Load part features for verification
        except Exception: pass

    def _load_part_features(self):
        """
        [2026-01-19] Pre-calculate Eye/Nose/Mouth embeddings from enrollment photos.
        This enables part-based verification to reject high-confidence misidentifications.
        """
        pb = os.path.join(self.local_media_path, "pic_bak")
        if not os.path.isdir(pb): return
        
        from init.mediapipe_handler import MediaPipeHandler
        mp_handler = MediaPipeHandler(max_num_faces=1) # Temporary instance
        
        try:
            pic_files = [f for f in os.listdir(pb) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not pic_files: return
            
            LOGGER.info(f"Loading Part Features for {len(pic_files)} users...")
            part_db = {}
            
            for f in tqdm(pic_files, desc="[Part Feature Gen]"):
                try:
                    # Filename: G07_..._Name.jpg
                    bn = os.path.splitext(f)[0]
                    staff_id = bn.split('_')[0] 
                    
                    img_path = os.path.join(pb, f)
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None: continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
                    boxes, _, points = mp_handler.detect(img_rgb)
                    if boxes is not None:
                        # Get part crops (Eye, Nose, Mouth)
                        # Note: We convert to PIL inside get_parts_crop if needed, or pass PIL
                        img_pil = Image.fromarray(img_rgb)
                        parts_tensors = get_parts_crop(img_pil, points[0])
                        
                        parts_emb = {}
                        for p_name, p_tensor in parts_tensors.items():
                            emb = self.resnet(p_tensor.unsqueeze(0)).detach().numpy()[0]
                            parts_emb[p_name] = emb
                        
                        part_db[staff_id] = parts_emb
                except Exception as e:
                    pass
                    
            self.state.part_features = part_db
            LOGGER.info(f"Loaded Part Features for {len(part_db)} users.")
        finally:
            mp_handler.close()

    def _update_profile_pictures(self):
        pb = os.path.join(self.local_media_path, "pic_bak")
        if not os.path.isdir(pb): self.state.profile_dict = {}; return
        lps = {}
        for f in os.listdir(pb):
            try:
                bn = os.path.splitext(f)[0]; pts = bn.split('_', 1)
                if len(pts) < 2: continue
                c, sk = pts[0].upper(), pts[1]
                if c not in lps or sk > lps[c]['sk']: lps[c] = {'path': os.path.join(pb, f), 'sk': sk}
            except Exception: continue
        self.state.profile_dict = {c: d['path'] for c, d in lps.items()}

    def _load_or_build_index(self, force_rebuild=False):
        if not self.state.features_dict or not any(self.state.features_dict.values()): return
        
        # 嘗試載入現有索引
        if not force_rebuild and self.state.ann_index.load():
            # [2026-01-18 Fix] 強化同步檢查：不只檢查數量，還檢查 ID 集合是否一致
            # 避免 "刪一增一" 導致數量相同但內容過期的問題
            
            # 1. 檢查數量
            current_count = sum(len(v) for k, v in self.state.features_dict.items() if k != 'id_name')
            if self.state.ann_index.index and self.state.ann_index.index.ntotal == current_count:
                # 2. 檢查 ID 內容 (確保索引內的 ID 都在目前的 features_dict 中)
                # self.state.ann_index.id_map 儲存了索引中每個向量對應的 Person ID
                cached_ids = set(self.state.ann_index.id_map)
                current_ids = set(k for k in self.state.features_dict.keys() if k != 'id_name')
                
                # 如果快取中的 ID 集合與目前的 ID 集合完全一致，才視為有效
                if cached_ids == current_ids:
                    return
                else:
                    LOGGER.warning("索引 ID 與目前檔案不符 (可能是刪除/新增導致數量巧合)，強制重建索引。")
            else:
                 LOGGER.info(f"索引數量不符 (Index: {self.state.ann_index.index.ntotal}, Files: {current_count})，重建索引。")
        
        # 重建索引
        self.state.ann_index.build(self.state.features_dict)

    def _get_valid_staff_ids(self):
        try:
            r = requests.get(f"{CONFIG['Server']['API_url']}/staffs/", timeout=10); r.raise_for_status()
            return {i["staff_id"] for i in r.json()}
        except Exception: return None

    def _update_descriptor_counts_on_server(self):
        vids = self._get_valid_staff_ids()
        if vids is None: return
        dp = os.path.join(self.local_media_path, "descriptors")
        if not os.path.isdir(dp): return
        cnts = defaultdict(int)
        for f in os.listdir(dp):
            if f.lower().endswith('.npy'):
                try: cnts[f.split('_')[0]] += 1
                except Exception: continue
        for sid in vids:
            try: requests.patch(f"{CONFIG['Server']['API_url']}/staffs/{sid}/", json={"descriptors": cnts.get(sid, 0)}, timeout=5)
            except Exception: pass

    def setup_cameras(self):
        ips = [CONFIG["cameraIP"]["in_camera"], CONFIG["cameraIP"]["out_camera"]]
        n = 2 if ips[0] != ips[1] else 1
        self.cameras = [CameraSystem(ip, i, n, self, CONFIG) for i, ip in enumerate(ips) if ip != "0"]
        for i in range(len(self.cameras)-(2-n)):
            if CONFIG.get("full_screen", False): self.cameras[i].win.showFullScreen()
            else: self.cameras[i].win.showNormal()
            self.cameras[i].win.activateWindow(); self.cameras[i].win.raise_()

    def update_inout_log(self):
        try:
            lj = API.Scan_today_log()
            for sid in lj.keys():
                if lj[sid]["state"] == "enter": self.state.check_time[sid] = [False, time.time()-100]
                elif sid in self.state.check_time:
                    self.state.check_time[sid] = [True, 0]
                    t = threading.Timer(5, clear_leave_employee, (self, sid, )); t.daemon = True; t.start()
        except Exception: pass
        t = threading.Timer(300, self.update_inout_log); t.daemon = True; t.start()

    def _safe_shutdown(self):
        if hasattr(self, 'speaker'): self.speaker.terminate()
        # [2026-01-13 Perf] Shutdown IO pool
        if hasattr(self, 'io_pool'): self.io_pool.shutdown(wait=False)
        QApplication.closeAllWindows()

if __name__ == "__main__":
    try: FaceRecognitionSystem().run()
    except Exception: os._exit(1)