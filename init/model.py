import json
import os
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import numba as nb
import torch
from init.log import LOGGER, PERF_LOGGER
from datetime import datetime
import pytz
import json
from PIL import Image
from init.function import crop_face_without_forehead, check_in_out_qrcode, check_in_out
from init.mediapipe_handler import MediaPipeHandler

@nb.jit
def cosine_similarity(vec1, vec2):
    """
    計算兩個向量之間的餘弦相似度。

    Parameters:
    vec1 (np.ndarray): 向量1
    vec2 (np.ndarray): 向量2

    Returns:
    float: 餘弦相似度
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# 載入設定檔
with open(os.path.join(os.path.dirname(__file__), "../config.json"), "r", encoding="utf-8") as json_file:
    CONFIG = json.load(json_file)
CAMERA = {0: "inCamera", 1: "outCamera"}
CAM_NAME_MAP = {0: "入口", 1: "出口"}
POTENTIAL_MISS_RATIO = 0.8
Z_SCORE_THRESHOLD = 1.5 

test_img = cv2.imread(os.path.join(
    os.path.dirname(__file__), "../other/test_img.jpg"))
test_img = cv2.resize(test_img, (224, 224))
tensor_test_img = torch.from_numpy(test_img).unsqueeze(0).permute(0, 3, 1, 2)


class Detector:
    """
    從系統中的即時畫面中偵測人臉，並觸發衣著（反光衣、安全帽）辨識功能。
    若為主畫面 (frame_num == 0)，會進行暖機與衣著辨識。
    """

    def __init__(self, frame_num, system):
        """
        使用 MediaPipe 替代 MTCNN。
        """
        self.system = system
        self.frame_num = frame_num
        self.TIMEZONE = pytz.timezone('Asia/Taipei')
        self.stop_threads = False
        self.last_face_time = 0
        self.last_no_face_log_time = 0
        self.clothe_time = [0, 0, 0]
        # 初始化 MediaPipe 處理器
        self.mp_handler = MediaPipeHandler()
        
        # [2026-02-04 Feature] QR Code Detector
        self.qr_detector = cv2.QRCodeDetector()
        self.last_qr_time = 0
        self.last_qr_data = ""
        self.qr_scan_interval = 1.0 # 1 FPS limit
        self.last_qr_scan_time = 0
        
        # [2026-02-03 Fix] 初始化衣著偵測旗標
        # 僅在入口攝影機 (frame_num == 0) 且設定開啟時執行
        # [2026-02-06 Fix] 若開啟 "Detection" (攔截)，即使 "Show" (顯示框) 關閉，也必須執行偵測，否則會因狀態全 False 而永久攔截
        self.do_clothes = (self.frame_num == 0 and (CONFIG.get("Clothes_show", False) or CONFIG.get("Clothes_detection", False)))
        
        threading.Thread(target=self.face_detector, daemon=True).start()

    def _is_entry_active(self):
        """
        [2026-02-06 Fix] 判斷當前是否為入口模式 (複製自 face_main.py 邏輯，供 Detector 使用)
        解決單鏡頭模式下，出口時段誤執行衣著偵測與阻斷的問題。
        """
        # 1. 判斷是否為單鏡頭
        ips = [CONFIG["cameraIP"]["in_camera"], CONFIG["cameraIP"]["out_camera"]]
        is_single_cam = (ips[0] == ips[1])
        
        # 雙鏡頭模式：看 frame_num
        if not is_single_cam:
            return self.frame_num == 0
            
        # 單鏡頭模式：看排程
        schedule_conf = CONFIG.get("Schedule", {})
        if not schedule_conf.get("enabled", False):
            return True # 無排程預設為入口 (從嚴)
            
        try:
            now_time = datetime.now().time()
            periods = schedule_conf.get("in_periods", [])
            if not periods:
                start_str = schedule_conf.get("in_start", "06:00")
                end_str = schedule_conf.get("in_end", "17:00")
                periods = [{"start": start_str, "end": end_str}]
            
            for period in periods:
                start_time = datetime.strptime(period.get("start", "00:00"), "%H:%M").time()
                end_time = datetime.strptime(period.get("end", "00:00"), "%H:%M").time()
                if start_time <= end_time:
                    if start_time <= now_time <= end_time: return True
                else:
                    if start_time <= now_time or now_time <= end_time: return True
            return False
        except:
            return True

    def face_detector(self):
        last_box = None
        last_points = None
        last_time = 0
        last_detection_time = 0
        DETECTION_INTERVAL = self.system.state.detection_interval
        dummy_input = tensor_test_img[0]

        while not self.stop_threads:
            now = time.time()
            if self.system.state.frame[self.frame_num] is not None:
                if now - last_detection_time > DETECTION_INTERVAL:
                    last_detection_time = now
                    
                    self.system.state.max_box[self.frame_num] = last_box
                    self.system.state.max_points[self.frame_num] = last_points
                    new_frame = self.system.state.frame[self.frame_num].copy()
                    
                    # [2026-02-03 Fix] 執行衣著偵測
                    # 使用局部變數暫存結果，偵測完成後再一次性更新全域狀態，避免 Race Condition
                    
                    # [2026-02-06 Fix] 動態判斷是否需要跑衣著偵測 (考慮單鏡頭排程)
                    # 必須同時滿足: 1. 初始化允許(do_clothes) 2. 當前是入口時段(_is_entry_active)
                    is_entry_now = self._is_entry_active()
                    should_detect_clothes = self.do_clothes and is_entry_now
                    
                    # [Fix] 確保變數始終初始化，避免 UnboundLocalError
                    current_clothes_detections = []
                    
                    if should_detect_clothes:
                        local_clothes_state = [False, False, False]
                        self.mask_frame, x_offset = self.apply_mask(new_frame)
                        try:
                            current_clothes_detections = self.clothes_detector(x_offset, local_clothes_state)
                        except Exception as e:
                            LOGGER.error(f"衣著偵測失敗: {e}")
                        
                        # [2026-02-03 Fix] 引入 Debounce 機制 (1.0秒)
                        # 若當前幀未偵測到，但 1 秒內曾偵測到，則保持 True，防止 UI 閃爍
                        now = time.time()
                        for i in range(3):
                            if not local_clothes_state[i]:
                                if (now - self.clothe_time[i]) < 1.0:
                                    local_clothes_state[i] = True
                        
                        # 原子性更新全域狀態
                        self.system.state.clothes = local_clothes_state
                    else:
                        # 若不偵測，是否該重置狀態？
                        if self.do_clothes: # 只有原本有開的才需要重置
                             self.system.state.clothes = [False, False, False]
                    
                    # [2026-02-04 Feature] QR Code Detection
                    # 與臉辨同時進行，但限制頻率 (1 FPS)
                    if CONFIG.get("qrcode_mode", False) and (now - self.last_qr_scan_time > self.qr_scan_interval):
                        self.last_qr_scan_time = now
                        # LOGGER.info("[DEBUG] Scanning for QR Code...") # Uncomment for debugging
                        try:
                            # 使用原始影像偵測，避免縮放導致無法讀取
                            # 嘗試轉灰階以提升偵測率
                            gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                            
                            qr_data, points, _ = self.qr_detector.detectAndDecode(gray_frame)
                            
                            if qr_data:
                                LOGGER.info(f"[QR Debug] Raw Data: {qr_data}") # Log raw data
                                
                                if qr_data != self.last_qr_data or (now - self.last_qr_time > 3.0):
                                    # Format: 111111G06 (Verification 6 digits + Staff ID)
                                    if len(qr_data) > 6 and qr_data[:6].isdigit():
                                        verification = qr_data[:6]
                                        staff_id = qr_data[6:]
                                        
                                        LOGGER.info(f"QR Code Detected: {qr_data} -> V:{verification}, ID:{staff_id}")
                                        self.last_qr_time = now
                                        self.last_qr_data = qr_data
                                        
                                        # Trigger Check-in/out
                                        check_in_out_qrcode(self.system, verification, staff_id, self.frame_num)
                                    else:
                                        LOGGER.warning(f"[QR Debug] Invalid Format: {qr_data}")
                        except Exception as e:
                            LOGGER.error(f"QR Scan Error: {e}") 
                            pass

                    new_high_res = None
                    if self.system.state.frame_high_res is not None and self.system.state.frame_high_res[self.frame_num] is not None:
                         new_high_res = self.system.state.frame_high_res[self.frame_num].copy()

                    box = None
                    points = None

                    # 1. 使用 MediaPipe 偵測
                    rgb_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                    boxes, _, landmarks = self.mp_handler.detect(rgb_frame)

                    if boxes is not None:
                        self.last_face_time = time.time()
                        x1, y1, x2, y2 = map(int, boxes[0])
                        points = landmarks[0].copy()
                        
                        # ROI 過濾與距離過濾
                        w_source = new_frame.shape[1]
                        close_N = 8 if CONFIG[CAMERA[self.frame_num]]["close"] else 6
                        roi_x1 = w_source // close_N
                        roi_x2 = (close_N - 1) * w_source // close_N
                        center_x = (x1 + x2) / 2
                        
                        if center_x < roi_x1 or center_x > roi_x2:
                            box = None
                        else:
                            face_width = x2 - x1
                            min_face_val = self.system.state.min_face[self.frame_num]
                            if face_width < (min_face_val * POTENTIAL_MISS_RATIO):
                                box = None
                            else:
                                box = [x1, y1, x2, y2]
                                
                                # [2026-02-06 Fix] 語音優先級控制 (阻斷邏輯) + 水平匹配驗證
                                # 若衣著不合格 (基於人臉位置的匹配)，阻斷後續品質檢查與辨識
                                # 防止 "請抬頭" (Comparison) 蓋過 "請著裝"
                                # [Fix] 只有在 "Clothes_detection" (攔截功能) 開啟時才執行阻斷
                                # [Fix] 且必須在入口時段 (is_entry_now)，避免出口被阻斷 (因為出口沒跑偵測，list為空)
                                if self.do_clothes and CONFIG.get("Clothes_detection", False) and is_entry_now:
                                    # 使用水平匹配檢查 (比全域檢查更準，比垂直檢查更穩)
                                    has_vest, has_helmet = self._match_clothes_to_face_horizontal(box, current_clothes_detections)
                                    
                                    if not (has_vest and has_helmet):
                                        # 1. 播放語音 (Priority 1)
                                        # 限流 2.0 秒，避免洗版
                                        if time.time() - self.last_no_face_log_time > 2.0:
                                            self.system.speaker.say("請正確著裝", "hint_clothes_block", priority=1)
                                            self.last_no_face_log_time = time.time()
                                        
                                        # 2. UI 提示
                                        self.system.state.hint_text[self.frame_num] = "請正確著裝"
                                        
                                        # 3. 關鍵阻斷：清除 frame_data，讓 Comparison 執行緒因無資料而閒置
                                        # 這樣它就不會執行 check_face_quality，自然不會發出 "請抬頭"
                                        self.system.state.frame_data[self.frame_num] = None
                                        
                                        # 4. 更新 UI 框 (顯示紅框/提示)
                                        self.system.state.max_box[self.frame_num] = box
                                        self.system.state.max_points[self.frame_num] = points
                                        self.system.state.gaze_status[self.frame_num] = None
                                        
                                        # 更新時間以免被視為 idle
                                        last_box = box
                                        last_points = points
                                        last_time = time.time()
                                        
                                        # 5. 跳過本幀後續處理
                                        continue

                                # [2026-01-11 Fix] 強制同步計算 Gaze，避免 Comparison 線程讀到舊的 State
                                # 在此時計算，mp_handler.last_results 必然對應當前這幀 new_frame
                                g_pass, g_msg, g_pose, g_ear = self.mp_handler.check_gaze(0)
                                # [2026-01-26 Fix] 將 Pose 和 EAR 一併打包入 gaze_status，確保原子性傳遞
                                self.system.state.gaze_status[self.frame_num] = (g_pass, g_msg, g_pose, g_ear)
                                self.system.state.head_pose[self.frame_num] = g_pose
                                
                                # [Deprecated] face_ear Global State is risky, use gaze_status tuple instead
                                if not hasattr(self.system.state, 'face_ear'):
                                    self.system.state.face_ear = {}
                                self.system.state.face_ear[self.frame_num] = g_ear
                                
                                # 存入系統以便 Comparison 呼叫 (雖然現在 Comparison 改讀 status 了)
                                self.system.mp_detectors[self.frame_num] = self.mp_handler
                    else:
                        self.system.state.gaze_status[self.frame_num] = None
                        self.system.state.head_pose[self.frame_num] = None

                    self.system.state.max_box[self.frame_num] = box
                    self.system.state.max_points[self.frame_num] = points
                    self.system.state.frame_mtcnn[self.frame_num] = new_frame
                    self.system.state.frame_mtcnn_high_res[self.frame_num] = new_high_res
                    
                    # [2026-01-11 Fix] 原子打包寫入，解決 Race Condition
                    g_status = self.system.state.gaze_status[self.frame_num]
                    self.system.state.frame_data[self.frame_num] = (new_frame, g_status, box, points)
                    
                    last_box = box
                    last_points = points
                    last_time = time.time()

            time.sleep(0.01)

    def clothes_detector(self, X_offset, state_buffer=None):
        """
        使用 YOLO 模型進行衣著偵測，標記安全帽與反光衣是否存在。
        [2026-02-06] Modified to return detection boxes for matching.

        Parameters:
        X_offset (int): 圖像遮罩偏移量，用來還原原始座標。
        state_buffer (list): 用於寫入結果的暫存列表，若為 None 則寫入全域狀態 (不建議)。

        Returns:
        detections (list): [(class_id, box_xyxy), ...]
        """
        # 偵測衣著（反光衣、安全帽）
        results = self.system.model_clothes(
            source=self.mask_frame,
            iou=0.45,
            conf=0.7,
            verbose=False
        )[0]

        detections = []
        cp_re = [0, 0, 0]
        for i, det in enumerate(results.boxes):
            class_id = int(det.cls)  # class_id: 0=反光衣, 2=安全帽
            box_xy = det.xywh[0]
            
            # Convert to absolute xyxy for matching
            x1 = int(box_xy[0] - box_xy[2]/2) + X_offset
            y1 = int(box_xy[1] - box_xy[3]/2)
            x2 = int(box_xy[0] + box_xy[2]/2) + X_offset
            y2 = int(box_xy[1] + box_xy[3]/2)
            
            detections.append((class_id, [x1, y1, x2, y2]))
            
            cp_re[class_id] = [x1, y1, x2, y2] # Legacy format just in case
            
            # [2026-02-03 Fix] 優先寫入 buffer，避免直接操作全域狀態造成閃爍
            if state_buffer is not None:
                state_buffer[class_id] = True
            else:
                self.system.state.clothes[class_id] = True
            
            self.clothe_time[class_id] = time.time()
            
        return detections

    def _match_clothes_to_face_horizontal(self, face_box, clothes_detections):
        """
        [2026-02-06 Feature] Face-Centric Clothes Verification (Horizontal Only)
        驗證偵測到的裝備是否在人臉的水平範圍內 (排除路人)。
        放棄垂直檢查以避免誤判 (如低頭、高帽)，改採寬鬆的水平鄰近檢查。
        """
        has_vest = False
        has_helmet = False
        
        fx1, fy1, fx2, fy2 = face_box
        face_cx = (fx1 + fx2) / 2
        face_w = fx2 - fx1
        
        # 允許偏差範圍：臉寬的 1.5 倍 (左右各 0.75)
        # 這足以涵蓋身體寬度，但能排除明顯在旁邊的路人
        threshold = face_w * 1.5
        
        for cls, box in clothes_detections:
            bx1, by1, bx2, by2 = box
            box_cx = (bx1 + bx2) / 2
            
            if abs(box_cx - face_cx) < threshold:
                if cls == 2: has_helmet = True
                elif cls == 0: has_vest = True
                        
        return has_vest, has_helmet

    def apply_mask(self, frame):
        """
        對輸入圖像應用水平遮罩（只保留中間區域）。
        [2026-02-06 Fix] 改用 35% 比例以對齊 UI 視覺遮罩，排除路人干擾。

        Parameters:
        frame (np.ndarray): 原始 BGR 圖像

        Returns:
        masked_frame (np.ndarray): 遮罩後的圖像區域
        X_offset (int): 遮罩區域的水平偏移量
        """
        # 遮罩處理，保留畫面中間的區域進行臉部偵測
        height, width, _ = frame.shape
        mask = np.zeros_like(frame)
        
        # [Fix] 使用 35% 比例 (左右各 17.5%)
        ratio = 0.35
        if CONFIG[CAMERA[self.frame_num]]["close"]:
            # 近距離模式可能需要寬一點? 先維持 50% 以防萬一，或者也設為 35%
            # 根據之前設定 (8 -> 75%)，這裡保守設為 0.5 (50%)
            ratio = 0.5
            
        center = width // 2
        half_w = int(width * ratio / 2)
        x1 = max(0, center - half_w)
        x2 = min(width, center + half_w)

        # 產生白色矩形遮罩
        cv2.rectangle(
            mask,
            (x1, 0),
            (x2, height),
            (255, 255, 255), -1
        )

        # 套用遮罩後回傳遮罩區域與偏移量
        masked_frame = cv2.bitwise_and(frame, mask)
        # 裁切出有效區域 (減少 YOLO 運算量)
        masked_frame = masked_frame[0:, x1:x2]
        
        return masked_frame, x1

    def equalize(self, img):
        """
        對輸入 BGR 圖像進行每通道的直方圖均衡化（增強對比）。

        Parameters:
        img (np.ndarray): 原始圖像

        Returns:
        equ_image (np.ndarray): 均衡化後的圖像
        """
        # 對 BGR 圖像做每個通道的直方圖均衡化
        b, g, r = cv2.split(img)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        equ_image = cv2.merge((b_eq, g_eq, r_eq))
        return equ_image

    def terminate(self):
        # 外部終止此執行緒
        self.stop_threads = True


class Comparison:
    """
    負責臉部向量比對與身份預測：
    - 最終決策者，統一控制辨識與顯示狀態。
    - 採用單次辨識成功即觸發的機制。
    - 引入顯示狀態保持機制，解決畫面閃爍問題。
    """

    def __init__(self, frame_num, system):
        self.system = system
        self.frame_num = frame_num
        self.stop_threads = False

        # 用於控制辨識成功後，人員名稱在畫面上停留的時間
        self.display_state = {'person_id': 'None', 'last_update': 0}
        self.last_recognition_time = 0

        self.last_api_trigger_time = {} # 記錄每個人員上次觸發API/語音的時間，用於防止短時間重複播報

        self.DISPLAY_STATE_HOLD_SECONDS = 2  # 辨識成功後，名稱顯示的持續時間
        self.CONFIDENCE_THRESHOLD = 0.7      # 可靠辨識的信賴度門檻 (員工)
        self.VISITOR_CONF_THRESHOLD = 0.5    # 訪客辨識的信賴度門檻 (低於此值為訪客)

        # --- 新增: 潛在辨識失敗分析與統計 ---
        self.width_stats = defaultdict(int)  # 統計人臉寬度分佈 (區間:次數)
        self.last_stats_log_time = 0         # 上次輸出統計表的時間
        self.potential_miss_ratio = POTENTIAL_MISS_RATIO # 潛在失敗判定門檻 (min_face * ratio)
        self.last_potential_miss_log_time = 0 # 上次記錄潛在失敗的時間 (限流用)
        self.hint_clear_time = 0             # 提示文字清除時間
        self.last_hint_speak_time = 0        # 上次播報提示語音的時間
        # ---------------------------------

        self.TIMEZONE = pytz.timezone('Asia/Taipei')

        threading.Thread(target=self.face_comparison, daemon=True).start()

    def _save_potential_miss_image(self, frame, width, threshold, camera_name, reason="Unknown"):
        """
        儲存潛在辨識失敗的截圖 (寬度介於意圖區間的人臉)。
        [2026-01-30] Added reason to filename.
        """
        try:
            today_str = datetime.now().strftime('%Y_%m_%d')
            time_str = datetime.now().strftime('%H;%M;%S')
            
            # 決定位置標記
            cam_tag = "Out" if "Out" in camera_name or "出口" in camera_name else "In"
            if "Cam" in camera_name: # Fallback for "Cam 0", "Cam 1"
                 cam_tag = "Out" if "1" in camera_name else "In"

            # 建立目錄 img_log/potential_miss/YYYY_MM_DD
            save_dir = os.path.join(os.getcwd(), "img_log", "potential_miss", today_str)
            os.makedirs(save_dir, exist_ok=True)
            
            # Sanitize reason string for filename
            safe_reason = reason.replace(" ", "_").replace("/", "-").replace(":", "").replace("(", "").replace(")", "").replace("<", "lt").replace(">", "gt")
            # Limit length to avoid OS limits
            if len(safe_reason) > 50: safe_reason = safe_reason[:50]
            
            # 檔名格式: HH;MM;SS_In_W{width}_Fail_{reason}.jpg
            filename = f"{time_str}_{cam_tag}_W{width}_Fail_{safe_reason}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            LOGGER.error(f"儲存潛在失敗截圖時發生錯誤: {e}")
            return None

    def _save_potential_miss_json(self, image_path, metrics, msg):
        """
        [2026-01-30 Feature] 為潛在失敗截圖產生搭配的 JSON 檔。
        """
        try:
            json_path = os.path.splitext(image_path)[0] + ".json"
            
            data = {
                "timestamp": datetime.now(self.TIMEZONE).isoformat(),
                "reason": msg,
                "metrics": metrics
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            LOGGER.error(f"儲存潛在失敗 JSON 時發生錯誤: {e}")

    def check_face_quality(self, box, points, frame_w, frame_h, gaze_status):
        """
        評估人臉品質並計算懲罰係數。
        
        Returns:
        quality_score (float): 1.0 代表完美，0.0 代表未達標
        msg (str): 詳細的評估訊息
        """
        # ---------------------------------------------------------
        # 1. 畫面置中檢查 (Center Alignment) - UI 需求
        # ---------------------------------------------------------
        face_center_x = (box[0] + box[2]) / 2
        frame_center_x = frame_w / 2
        offset = abs(face_center_x - frame_center_x)
        limit_offset = frame_w * 0.15 # 允許偏離 15%
        margin = 5
        
        # [2026-01-30 Fix] Initialize current_ear to prevent UnboundLocalError if face_w <= 100
        current_ear = 1.0

        metrics = {
            'center_offset_px': float(offset),
            'center_limit_px': float(limit_offset),
            'face_width_px': float(box[2] - box[0]),
            'visibility_margin': float(margin),
            'gaze_passed': False,
            'gaze_msg': 'Init',
            'ear': 1.0,
            'v_ratio': 0.0,
            'roll_angle': 0.0, # Not strictly calculated here, but could be added if needed
            'pitch_check': 'Pass',
            'yaw_check': 'Pass'
        }

        if offset > limit_offset:
            return 0.0, f"未置中 (偏離 {offset:.1f}px > 容許 {limit_offset:.1f}px)", metrics

        # ---------------------------------------------------------
        # 2. 特徵點完整性檢查 (Visibility) - 完整性需求
        # ---------------------------------------------------------
        for i, p in enumerate(points):
            if p[0] < margin or p[0] > frame_w - margin or \
               p[1] < margin or p[1] > frame_h - margin:
                 return 0.0, f"特徵點被切除/遮擋 (點{i}座標 {p} 超出邊界)", metrics

        # ---------------------------------------------------------
        # 2.5 夕陽/強光檢查 (Sunset/Overexposure) - [2026-02-07 Feature]
        # ---------------------------------------------------------
        from init.function import is_sunset_condition # Local import to avoid circular dependency
        
        # 由於此檢查需要 crop ROI，為了效能，只在人臉足夠大時執行
        face_w = max(10, box[2] - box[0])
        if face_w > 100:
            frame_to_use = self.system.state.frame_mtcnn[self.frame_num]
            if frame_to_use is not None:
                if is_sunset_condition(frame_to_use, box, points):
                    return 0.0, "光線直射 (Sunset Mode)", metrics

        # ---------------------------------------------------------
        # 3. 3D 姿態與視線檢查 (Gaze & Pose Check) - 核心邏輯
        # ---------------------------------------------------------
        # [2026-01-11 Fix] 直接使用傳入的同步狀態，解決影像與判定錯位問題
        if face_w > 100:
            if gaze_status:
                # [2026-01-26 Fix] 兼容擴充後的 gaze_status (4 elements: pass, msg, pose, ear)
                is_looking = gaze_status[0]
                gaze_msg = gaze_status[1]
                
                # 預先提取 EAR，優先使用原子打包數據
                current_ear = 1.0
                if len(gaze_status) >= 4:
                    current_ear = gaze_status[3]
                elif hasattr(self.system.state, 'face_ear'): # Fallback for backward compatibility
                    current_ear = self.system.state.face_ear.get(self.frame_num, 1.0)

                metrics['gaze_passed'] = is_looking
                metrics['gaze_msg'] = gaze_msg
                metrics['ear'] = float(current_ear)

                if not is_looking:
                    return 0.0, f"{gaze_msg}", metrics
            else:
                # [2026-01-11 Fix] 若無 Gaze 狀態 (可能因 Race Condition 被清空)，嚴格禁止放行
                return 0.0, "Gaze Status Missing", metrics

        # ---------------------------------------------------------
        # 3.1 幾何比例檢查 (Geometry Check) - 低頭防禦
        # ---------------------------------------------------------
        # [2026-01-22 Fix] 防止極端低頭導致特徵崩壞誤判 (V-Ratio < 0.55)
        # Points: 0:LE, 1:RE, 2:Nose, 3:LM, 4:RM
        eye_y = (points[0][1] + points[1][1]) / 2
        nose_y = points[2][1]
        mouth_y = (points[3][1] + points[4][1]) / 2
        
        eye_nose_dist = nose_y - eye_y
        nose_mouth_dist = mouth_y - nose_y
        
        if eye_nose_dist > 0:
            v_ratio = nose_mouth_dist / eye_nose_dist
            metrics['v_ratio'] = float(v_ratio)
            # 正常值: 0.8 ~ 1.2, 低頭測試照: 0.18 ~ 0.40
            
            # 1. 極端低頭過濾 (絕對死線)
            # 殺死 9 張極端低頭測試照 (V < 0.35)
            if v_ratio < 0.35:
                metrics['pitch_check'] = 'Fail (Extreme Low)'
                return 0.0, f"低頭 (V-Ratio: {v_ratio:.2f} < 0.35)", metrics
            
            # 2. 低頭+遮眼 Combo 過濾 (0.35 <= V < 0.42)
            # [2026-01-22] 針對灰色地帶進行補刀
            # - 蔡準庭帽子照 (V=0.403, EAR=0.213) -> 符合雙重條件 -> KILL
            # - 楊昌裕 (V=0.47, EAR=0.10) -> V正常 -> PASS
            # - 林文明 (V=0.40, EAR=0.26) -> EAR正常 -> PASS
            if v_ratio < 0.42:
                # [2026-01-26 Refactor] Use pre-extracted EAR
                if current_ear < 0.22:
                    metrics['pitch_check'] = 'Fail (Combo Low+Cover)'
                    return 0.0, f"低頭/遮眼 (V {v_ratio:.2f}<0.42 & EAR {current_ear:.2f}<0.22)", metrics

        # ---------------------------------------------------------
        # 3.2 閉眼檢查 (Eye Closure Check) - [2026-01-26 Fix]
        # ---------------------------------------------------------
        # 根據測試，閉眼誤判照 EAR=0.0694，小眼(楊昌裕) EAR=0.0837。
        # 設定底層安全門檻 0.05 (極端閉眼)。
        # 中間地帶 (0.05~0.10) 交由 mp_handler 的 Combo Check 處理。
        if current_ear < 0.075:
            return 0.0, f"眼睛閉合 (EAR: {current_ear:.4f} < 0.075)", metrics

        # ---------------------------------------------------------
        # 4. 臉部區域清晰度檢查 (ROI Blur Detection)
        # ---------------------------------------------------------
        # [2026-01-11] 實驗數據：誤判糊臉=7.1, 正常辨識平均=20.6
        # [2026-01-18 Disabled by User Request]
        # try:
        #     x1, y1, x2, y2 = map(int, box)
        #     x1, y1 = max(0, x1), max(0, y1)
        #     x2, y2 = min(frame_w, x2), min(frame_h, y2)
        #     
        #     face_roi = self.system.state.frame_mtcnn[self.frame_num][y1:y2, x1:x2]
        #     if face_roi.size > 0:
        #         gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        #         blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        #         
        #         if blur_score < 13.0:
        #             return 0.0, f"影像模糊 (ROI Score:{blur_score:.1f} < 13.0)"
        # except Exception as e:
        #     LOGGER.error(f"清晰度檢查失敗: {e}")

        # 通過所有檢查
        return 1.0, "Pass", metrics

    def _update_display_state(self, person_id):
        """更新當前顯示的人員ID和時間"""
        self.display_state['person_id'] = person_id
        self.display_state['last_update'] = time.time()
        self.system.state.same_class[self.frame_num] = person_id

    def face_comparison(self):
        """
        執行臉部比對的核心迴圈。
        - 提取人臉特徵。
        - 與資料庫比對並計算信賴度。
        - 如果信賴度超過門檻，則觸發成功事件。
        - 管理UI顯示狀態，並移除信賴度分數的顯示。
        - 引入 Z-Score 離群值分析，以提高在高相似度誤判情況下的準確性。
        - 新增辨識品質評級 (可靠/模糊/低信賴度) 至日誌。
        """
        last_warmup_time = 0
        dummy_input = tensor_test_img

        while not self.stop_threads:
            # 動態調整頻率
            time.sleep(self.system.state.comparison_interval)
            now = time.time()
            
            # 清除過期的 UI 提示
            if now > self.hint_clear_time:
                self.system.state.hint_text[self.frame_num] = ""

            # 清除畫面上的人員名稱（如果超過顯示時間）
            if self.display_state['person_id'] != 'None' and \
               now - self.display_state['last_update'] > self.DISPLAY_STATE_HOLD_SECONDS:
                self._update_display_state('None')

            # [2026-01-11 Fix] 原子讀取打包數據
            # 確保 影像(frame), 狀態(gaze), 位置(box) 來自同一時間點 (Snapshot)
            data_package = self.system.state.frame_data[self.frame_num]
            if data_package is None:
                continue
                
            _frame, _gaze_status, _box, _points = data_package
            
            # 使用解包出來的 frame，而不是去讀可能已經被覆蓋的 system.state.frame_mtcnn
            self.system.state.frame_mtcnn[self.frame_num] = _frame # 為了相容其他可能讀取這欄位的地方(如UI?)
            
            if _box is None or _points is None or _frame is None:
                continue
            
            # 取得畫面尺寸 (用於置中與邊界檢查)
            frame_curr = _frame
            frame_h, frame_w, _ = frame_curr.shape
            
            camera_name = CAM_NAME_MAP.get(self.frame_num, f"Cam {self.frame_num}")

            # 檢查臉部大小是否足夠
            face_width = _box[2] - _box[0]
            min_face_threshold = self.system.state.min_face[self.frame_num]
            
            # --- 統計: 記錄人臉寬度分佈 (每 10px 為一個區間) ---
            width_bin = (face_width // 10) * 10
            self.width_stats[f"{width_bin}-{width_bin+9}"] += 1
            
            # 定期輸出統計摘要 (每分鐘一次，方便即時驗證)
            if now - self.last_stats_log_time > 60:
                stats_str = ", ".join([f"{k}: {v}" for k, v in sorted(self.width_stats.items())])
                LOGGER.info(f"[統計] [{camera_name}] 過去一分鐘人臉寬度分佈: {stats_str}")
                self.width_stats.clear() # 重置統計
                self.last_stats_log_time = now
            # -----------------------------------------------

            # [2026-01-08 夜間模式全域過濾]
            # 若為夜間 (18:00-06:00)，在進行任何品質或大小檢查前，先驗證「像不像人」
            current_hour = datetime.now(self.TIMEZONE).hour
            is_night_mode = (current_hour >= 18 or current_hour < 6)
            
            # 提取特徵向量 (為了夜間檢查或後續辨識)
            current_face_vec = None
            try:
                frame_to_use = _frame
                box_to_use = list(_box)
                points_to_use = _points.copy()
                frame_image = Image.fromarray(cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2RGB))
                img_cropped = crop_face_without_forehead(frame_image, box_to_use, points_to_use)
                face_embedding_list = self.system.resnet(img_cropped.unsqueeze(0))

                if face_embedding_list is not None and len(face_embedding_list) > 0:
                    current_face_vec = face_embedding_list[0].detach().numpy()
            except Exception as e:
                LOGGER.error(f"[{camera_name}] 特徵提取失敗: {e}")
                pass

            # 夜間強力過濾
            if is_night_mode and current_face_vec is not None:
                if self.system.state.ann_index and self.system.state.ann_index.index is not None and self.system.state.ann_index.index.ntotal > 0:
                    dists, _ = self.system.state.ann_index.search(current_face_vec, k=1)
                    if dists[0] < 0.4:
                        continue

            # [2026-01-11] 判斷是否處於 "辨識成功後的顯示保持期"
            is_staff_displaying = (
                self.display_state['person_id'] != 'None' and 
                self.display_state['person_id'] != '__VISITOR__' and
                (now - self.display_state['last_update'] < self.DISPLAY_STATE_HOLD_SECONDS)
            )

            if face_width < min_face_threshold:
                if self.display_state['person_id'] != 'None' and not is_staff_displaying:
                    self._update_display_state('None')

                potential_threshold = min_face_threshold * self.potential_miss_ratio
                
                if face_width >= potential_threshold:
                    if now - self.last_potential_miss_log_time > 3:
                        snapshot = _frame
                        saved_path = "無影像"
                        if snapshot is not None:
                            # [2026-01-30] Pass reason="SmallFace"
                            saved_path = self._save_potential_miss_image(snapshot, face_width, min_face_threshold, camera_name, reason="SmallFace")
                            
                        LOGGER.info(f"[{camera_name}][潛在失敗] 偵測到人臉但過小 (寬度: {face_width}) - 已存檔: {saved_path}")
                        self.last_potential_miss_log_time = now
                        
                        if not is_staff_displaying:
                            self.system.state.hint_text[self.frame_num] = "請靠近鏡頭"
                            self.hint_clear_time = now + 2.0
                            self.system.speaker.say("請靠近鏡頭", "hint_closer", priority=2)
                
                continue

            if face_width >= CONFIG["max_face"]:
                if not is_staff_displaying:
                    self.system.state.hint_text[self.frame_num] = "請稍微後退"
                    self.system.speaker.say("請稍微後退", "hint_move_back", priority=2)
                    self.hint_clear_time = time.time() + 1.5
                    if self.display_state['person_id'] != 'None':
                        self._update_display_state('None')
                continue

            # 檢查人臉品質 (同步版)
            quality_score, quality_msg, quality_metrics = self.check_face_quality(_box, _points, frame_w, frame_h, _gaze_status)
            
            if quality_score == 0.0:
                 if is_staff_displaying:
                     continue # 免死金牌

                 LOGGER.info(f"[{camera_name}][品質過濾] {quality_msg}")
                 
                 # [2026-01-30 Feature] 潛在失敗數據收集 (大臉但被品質過濾)
                 if face_width >= min_face_threshold and now - self.last_potential_miss_log_time > 1.0:
                     try:
                         snapshot = _frame
                         if snapshot is not None:
                             saved_path = self._save_potential_miss_image(snapshot, face_width, min_face_threshold, camera_name, reason=quality_msg)
                             # 產生搭配的 JSON
                             if saved_path:
                                 self._save_potential_miss_json(saved_path, quality_metrics, quality_msg)
                             
                             LOGGER.info(f"[{camera_name}][品質失敗收集] 寬度 {face_width} 但品質未過 - 已存檔")
                             self.last_potential_miss_log_time = now
                     except Exception as e:
                         LOGGER.error(f"Save potential miss (quality) failed: {e}")

                 if "低頭" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請抬頭"
                     self.system.speaker.say("請抬頭", "hint_look_up", priority=2)
                 elif "抬頭" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請低頭"
                     self.system.speaker.say("請低頭", "hint_look_down", priority=2)
                 elif "未置中" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請站到中間"
                     self.system.speaker.say("請站到中間", "hint_center", priority=2)
                 elif "斜視" in quality_msg or "未正視" in quality_msg or "側臉" in quality_msg or "影像模糊" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請正視鏡頭"
                     self.system.speaker.say("請正視鏡頭", "hint_look_straight", priority=2)
                 elif "光線直射" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "光線直射 請遮擋"
                     self.system.speaker.say("光線直射請遮擋", "hint_sunset", priority=2)
                 else:
                     self.system.state.hint_text[self.frame_num] = "請對準鏡頭"
                     self.system.speaker.say("請對準鏡頭", "hint_occlusion", priority=2)
                     
                 self.hint_clear_time = now + 1.0
                 continue

            if current_face_vec is None:
                try:
                    comparison_start_time = time.monotonic()
                    frame_to_use = _frame
                    box_to_use = list(_box)
                    points_to_use = _points.copy()
                    frame_image = Image.fromarray(cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2RGB))
                    img_cropped = crop_face_without_forehead(frame_image, box_to_use, points_to_use)
                    face_embedding_list = self.system.resnet(img_cropped.unsqueeze(0))
                    if face_embedding_list is None or len(face_embedding_list) == 0:
                        continue
                    current_face_vec = face_embedding_list[0].detach().numpy()
                except Exception as e:
                    LOGGER.error(f"[ERROR][{camera_name}] 臉部特徵提取失敗: {e}")
                    continue
            else:
                 comparison_start_time = time.monotonic()

            try:
                if self.system.state.ann_index is None or self.system.state.ann_index.index is None or self.system.state.ann_index.index.ntotal == 0:
                    predicted_id = "None"
                    confidence = 0.0
                    z_score = 0.0
                    raw_confidence = 0.0
                    part_msg = ""
                else:
                    distances, faiss_person_ids = self.system.state.ann_index.search(current_face_vec, k=self.system.state.ann_index.index.ntotal)
                    if faiss_person_ids is None or len(faiss_person_ids) == 0:
                        predicted_id = "None"; confidence = 0.0; z_score = 0.0; raw_confidence = 0.0; part_msg = ""
                    else:
                        top_k_similarities = np.array(distances)
                        
                        # 1. Phase 1: Filter Candidates (Confidence >= 0.7 AND Z >= 1.5)
                        candidates = []
                        
                        # Calculate population stats from All Candidates (Option SMALL Logic)
                        if len(top_k_similarities) > 1:
                            mean_score = np.mean(top_k_similarities)
                            std_dev_score = np.std(top_k_similarities)
                        else:
                            mean_score = 0
                            std_dev_score = 0
                        
                        for i, pid in enumerate(faiss_person_ids):
                            s_raw = distances[i]
                            s_final = s_raw * quality_score
                            
                            z = (s_raw - mean_score) / std_dev_score if std_dev_score > 0 else 0
                            
                            # Strict Filter: Must pass BOTH thresholds
                            if s_final >= self.CONFIDENCE_THRESHOLD and z >= Z_SCORE_THRESHOLD:
                                candidates.append({
                                    'id': pid, 
                                    'raw': s_raw, 
                                    'conf': s_final, 
                                    'z': z
                                })
                        
                        # Set Default Winner (Top 1) for fallback/logging
                        best_match_id = faiss_person_ids[0]
                        raw_confidence = distances[0]
                        confidence = raw_confidence * quality_score
                        z_score = (raw_confidence - mean_score) / std_dev_score if std_dev_score > 0 else 0
                        part_msg = ""
                        
                        # [2026-02-01 Feature] Gap Check for Ambiguity Rejection
                        # 攔截高分誤判 (High Confidence False Positive)
                        gap = 0.0
                        if len(distances) > 1:
                            gap = float(distances[0]) - float(distances[1])
                            
                        # Dynamic Threshold Formula
                        # 如果信心度極高 (>0.80)，容忍較小的 Gap (0.02)
                        # 否則需要較大的 Gap (0.03) 以確保安全
                        gap_threshold = 0.02 if confidence > 0.80 else 0.03
                        
                        if gap < gap_threshold:
                             LOGGER.info(f"[{camera_name}][Gap過濾] 分數過於接近 (Gap: {gap:.4f} < {gap_threshold}) - 拒絕辨識")
                             
                             # [2026-01-30 Feature] 潛在失敗數據收集 (Gap Fail)
                             if face_width >= min_face_threshold and now - self.last_potential_miss_log_time > 1.0:
                                 try:
                                     snapshot = _frame
                                     if snapshot is not None:
                                         reason_str = f"Gap_Fail_{gap:.4f}"
                                         saved_path = self._save_potential_miss_image(snapshot, face_width, min_face_threshold, camera_name, reason=reason_str)
                                         if saved_path:
                                             self._save_potential_miss_json(saved_path, quality_metrics, f"Gap Fail: {gap:.4f}")
                                         self.last_potential_miss_log_time = now
                                 except: pass
                             
                             continue

                        # [2026-01-24 Feature] 記錄 Top-5 搜尋結果供除錯重現
                        top5_results = []
                        # Log top 5 only for debugging
                        log_k = min(5, len(faiss_person_ids))
                        for i in range(log_k):
                            pid = faiss_person_ids[i]
                            s_raw = distances[i]
                            z = (s_raw - mean_score) / std_dev_score if std_dev_score > 0 else 0
                            top5_results.append({
                                "rank": i + 1,
                                "id": pid,
                                "score": float(s_raw),
                                "z_score": float(z)
                            })
                        
                        # 2. Single Stage Decision (Option SMALL)
                        # No T-Zone re-ranking. Just pick the best candidate that passed filters.
                        t_zone_applied = False
                        t_zone_score = None
                        
                        if candidates:
                            # Candidates are populated in order of FAISS result (descending similarity)
                            # So candidates[0] is the best match that passed filters.
                            winner = candidates[0]
                            best_match_id = winner['id']
                            raw_confidence = winner['raw']
                            confidence = winner['conf']
                            z_score = winner['z']
                        
                        predicted_id = best_match_id
                        
                        # [2026-01-24 Feature] 建立完整的 Snapshot Metadata (供離線重現測試)
                        if current_face_vec is not None:
                            meta = {
                                "timestamp": datetime.now(self.TIMEZONE).isoformat(),
                                "predicted_id": best_match_id,
                                "full_score": float(confidence),
                                "z_score": float(z_score),
                                "quality_score": float(quality_score),
                                "quality_metrics": quality_metrics, # [2026-01-30] Add metrics
                                "t_zone_score": None,
                                "top5": top5_results,
                                "embedding": current_face_vec.tolist()
                            }
                            self.system.state.success_metadata[self.frame_num] = meta

            except Exception as e:
                LOGGER.error(f"[ERROR][{camera_name}] 預測失敗: {e}")
                continue

            # 標記每一次辨識事件（無論成功與否）
            log_time = datetime.now(
                self.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')

            staff_name = self.system.state.features_dict.get("id_name", {}).get(predicted_id, "未知")
            
            # [2026-01-11 Fix] 補回遺漏的 Log 訊息定義
            quality_rating = "Low Confidence"
            if confidence >= self.CONFIDENCE_THRESHOLD:
                if z_score >= Z_SCORE_THRESHOLD:
                    quality_rating = "Reliable"
                else:
                    quality_rating = "Ambiguous (Low Z)"
            
            # Log output to file
            log_msg = f"[{camera_name}] ID: {predicted_id} ({staff_name}), Score: {confidence:.2f} (Raw:{raw_confidence:.2f}), Z: {z_score:.2f}, Q: {quality_score:.2f} [{quality_rating}]{part_msg}"
            LOGGER.info(log_msg)

            if predicted_id != "None" and confidence >= self.CONFIDENCE_THRESHOLD and z_score >= Z_SCORE_THRESHOLD:
                if self.system.state.same_class[self.frame_num] != predicted_id:
                    self._update_display_state(predicted_id)
                    
                    # 辨識成功，播放音效與打卡
                    # [2026-01-08 Refactor] 統一使用新的打卡邏輯
                    # 傳入 check_in_out 進行防抖與方向判斷
                    # [2026-01-20 Fix] 傳入 confidence 供日誌記錄
                    check_in_out(self.system, staff_name, predicted_id, self.frame_num, self.system.n_camera < 2, confidence)
                    
            elif predicted_id != "None" and confidence >= self.VISITOR_CONF_THRESHOLD:
                # 訪客邏輯 (分數介於 0.5 ~ 0.7)
                # 為了避免員工側臉被誤判為訪客，這裡可以加一些限制，或者直接顯示訪客
                # 目前設定: 只要不是 Low Confidence 且沒過員工門檻，就視為訪客
                # [2026-01-29 Fix] 擴大模糊區間: 0.5 ~ 0.7 視為 Ambiguous/Ignore，不顯示訪客
                # 除非有特殊需求，否則不輕易跳出訪客，以免干擾員工
                pass
                # if self.system.state.same_class[self.frame_num] != '__VISITOR__':
                #     self._update_display_state('__VISITOR__')
            else:
                # Low Confidence or None
                pass

    def terminate(self):
        self.stop_threads = True
