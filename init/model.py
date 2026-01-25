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
from function import crop_face_without_forehead
from init.mediapipe_handler import MediaPipeHandler # 新增

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
        threading.Thread(target=self.face_detector, daemon=True).start()

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
                                # [2026-01-11 Fix] 強制同步計算 Gaze，避免 Comparison 線程讀到舊的 State
                                # 在此時計算，mp_handler.last_results 必然對應當前這幀 new_frame
                                g_pass, g_msg, g_pose, g_ear = self.mp_handler.check_gaze(0)
                                self.system.state.gaze_status[self.frame_num] = (g_pass, g_msg)
                                self.system.state.head_pose[self.frame_num] = g_pose
                                
                                # [2026-01-22 Fix] Store EAR
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

    def clothes_detector(self, X_offset):
        """
        使用 YOLO 模型進行衣著偵測，標記安全帽與反光衣是否存在。

        Parameters:
        X_offset (int): 圖像遮罩偏移量，用來還原原始座標。
        """
        # 偵測衣著（反光衣、安全帽）
        results = self.system.model_clothes(
            source=self.mask_frame,
            iou=0.45,
            conf=0.2,
            verbose=False
        )[0]

        cp_re = [0, 0, 0]
        for i, det in enumerate(results.boxes):
            class_id = int(det.cls)  # class_id: 0=反光衣, 2=安全帽
            box_xy = det.xywh[0]
            cp_re[class_id] = [
                box_xy[0] + X_offset,
                box_xy[1],
                box_xy[0] + X_offset + box_xy[2],
                box_xy[0] + box_xy[3]
            ]
            self.system.state.clothes[class_id] = True
            self.clothe_time[class_id] = time.time()

    def apply_mask(self, frame):
        """
        對輸入圖像應用水平遮罩（只保留中間區域）。

        Parameters:
        frame (np.ndarray): 原始 BGR 圖像

        Returns:
        masked_frame (np.ndarray): 遮罩後的圖像區域
        X_offset (int): 遮罩區域的水平偏移量
        """
        # 遮罩處理，保留畫面中間的區域進行臉部偵測
        height, width, _ = frame.shape
        mask = np.zeros_like(frame)
        close_N = 6 # 預設保留中間 4/6 (約 66%)，原為 3 (保留 33%)
        if CONFIG[CAMERA[self.frame_num]]["close"]:
            close_N = 8 # 近距離模式保留中間 6/8 (約 75%)，原為 4 (保留 50%)

        # 產生白色矩形遮罩
        cv2.rectangle(
            mask,
            (width // close_N, 0),
            ((close_N - 1) * width // close_N, height),
            (255, 255, 255), -1
        )

        # 套用遮罩後回傳遮罩區域與偏移量
        masked_frame = cv2.bitwise_and(frame, mask)
        masked_frame = frame[0:, width //
                             close_N: (close_N - 1) * width // close_N]
        return masked_frame, width // close_N

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

    def _save_potential_miss_image(self, frame, width, threshold, camera_name):
        """
        儲存潛在辨識失敗的截圖 (寬度介於意圖區間的人臉)。
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
            
            # 檔名格式: HH;MM;SS_In_W{width}_T{threshold}.jpg
            filename = f"{time_str}_{cam_tag}_W{width}_T{threshold}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            LOGGER.error(f"儲存潛在失敗截圖時發生錯誤: {e}")
            return None

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
        
        if offset > limit_offset:
            return 0.0, f"未置中 (偏離 {offset:.1f}px > 容許 {limit_offset:.1f}px)"

        # ---------------------------------------------------------
        # 2. 特徵點完整性檢查 (Visibility) - 完整性需求
        # ---------------------------------------------------------
        margin = 5
        for i, p in enumerate(points):
            if p[0] < margin or p[0] > frame_w - margin or \
               p[1] < margin or p[1] > frame_h - margin:
                 return 0.0, f"特徵點被切除/遮擋 (點{i}座標 {p} 超出邊界)"

        # ---------------------------------------------------------
        # 3. 3D 姿態與視線檢查 (Gaze & Pose Check) - 核心邏輯
        # ---------------------------------------------------------
        # [2026-01-11 Fix] 直接使用傳入的同步狀態，解決影像與判定錯位問題
        face_w = max(10, box[2] - box[0])
        if face_w > 100:
            if gaze_status:
                is_looking, gaze_msg = gaze_status
                if not is_looking:
                    return 0.0, f"{gaze_msg}"
            else:
                # [2026-01-11 Fix] 若無 Gaze 狀態 (可能因 Race Condition 被清空)，嚴格禁止放行
                return 0.0, "Gaze Status Missing"

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
            # 正常值: 0.8 ~ 1.2, 低頭測試照: 0.18 ~ 0.40
            
            # 1. 極端低頭過濾 (絕對死線)
            # 殺死 9 張極端低頭測試照 (V < 0.35)
            if v_ratio < 0.35:
                return 0.0, f"低頭 (V-Ratio: {v_ratio:.2f} < 0.35)"
            
            # 2. 低頭+遮眼 Combo 過濾 (0.35 <= V < 0.42)
            # [2026-01-22] 針對灰色地帶進行補刀
            # - 蔡準庭帽子照 (V=0.403, EAR=0.213) -> 符合雙重條件 -> KILL
            # - 楊昌裕 (V=0.47, EAR=0.10) -> V正常 -> PASS
            # - 林文明 (V=0.40, EAR=0.26) -> EAR正常 -> PASS
            if v_ratio < 0.42:
                current_ear = 1.0
                if hasattr(self.system.state, 'face_ear'):
                    current_ear = self.system.state.face_ear.get(self.frame_num, 1.0)
                
                if current_ear < 0.22:
                    return 0.0, f"低頭/遮眼 (V {v_ratio:.2f}<0.42 & EAR {current_ear:.2f}<0.22)"

        # ---------------------------------------------------------
        # 3.2 閉眼檢查 (Eye Closure Check) - [2026-01-26 Fix]
        # ---------------------------------------------------------
        # 根據測試，閉眼誤判照 EAR=0.0694，小眼(楊昌裕) EAR=0.0837。
        # 設定精確門檻 0.075，以在過濾誤判的同時保留小眼特徵。
        current_ear = 1.0
        if hasattr(self.system.state, 'face_ear'):
            current_ear = self.system.state.face_ear.get(self.frame_num, 1.0)
        
        if current_ear < 0.075:
            return 0.0, f"眼睛閉合 (EAR: {current_ear:.4f} < 0.075)"

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
        return 1.0, "Pass"

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
                            saved_path = self._save_potential_miss_image(snapshot, face_width, min_face_threshold, camera_name)
                            
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
            quality_score, quality_msg = self.check_face_quality(_box, _points, frame_w, frame_h, _gaze_status)
            
            if quality_score == 0.0:
                 if is_staff_displaying:
                     continue # 免死金牌

                 LOGGER.info(f"[{camera_name}][品質過濾] {quality_msg}")
                 
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
            is_reliable = confidence >= self.CONFIDENCE_THRESHOLD and z_score >= Z_SCORE_THRESHOLD
            is_visitor = confidence < self.VISITOR_CONF_THRESHOLD
            
            rating_str = "模糊 (Ambiguous)"
            if is_reliable:
                rating_str = "可靠 (Reliable)"
            elif is_visitor:
                rating_str = "訪客 (Visitor)"
            
            quality_info = ""
            if quality_score < 1.0:
                quality_info = f" (原:{raw_confidence:.2%}, 品質:{quality_score:.2f} {quality_msg})"
            
            log_message = (
                f"{log_time} [辨識事件][{camera_name}] 偵測到 {staff_name} (ID: {predicted_id}), "
                f"信賴度: {confidence:.2%}{quality_info}, Z-Score: {z_score:.2f}, Width: {face_width} px [評級: {rating_str}]"
            )
            
            # [2026-01-23 Fix] 統一記錄所有辨識事件 (包含可靠/訪客/模糊)，解決報表數據遺失問題
            LOGGER.info(log_message)
            
            if is_reliable:
                person_id = predicted_id
                self.system.state.hint_text[self.frame_num] = "" # 畫面優先
                
                # [2026-01-24 Fix] 原子打包 success_snapshot = (frame, metadata)
                # 確保 CameraSystem 讀取時，frame 和 metadata 一定來自同一次辨識
                try:
                    snapshot_frame = _frame.copy()
                    snapshot_meta = self.system.state.success_metadata[self.frame_num]
                    self.system.state.success_snapshot[self.frame_num] = (snapshot_frame, snapshot_meta)
                    # 保留舊欄位以相容其他程式碼
                    self.system.state.success_frame[self.frame_num] = snapshot_frame
                except:
                    pass

                speaker = self.system.speaker
                # [2026-01-20 Logic] 移除 10s/2s 冷卻，改由 Speaker 類別統一控管播放狀態
                # 只要 Speaker 空閒，就會播放；若 Speaker 忙碌 (P1)，則此次觸發會被 Speaker 丟棄。
                if True:
                    self.system.state.same_people[self.frame_num] = confidence
                    self.system.state.same_zscore[self.frame_num] = z_score
                    self.system.state.same_width[self.frame_num] = face_width

                self._update_display_state(person_id)
                self.last_recognition_time = now
            elif is_visitor:
                 # LOGGER.info(log_message) # 已統一移至上方
                 # 標記為訪客，但不觸發打卡
                 self.system.state.same_people[self.frame_num] = 0.0
                 self._update_display_state("__VISITOR__")
            else:
                 # Ambiguous Case: 信賴度不足 (0.5~0.7)，記錄 Log 以供除錯
                 # LOGGER.info(log_message) # 已統一移至上方
                 pass

            if time.time() - last_warmup_time > 10 and self.frame_num == 0:
                try:
                    _ = self.system.resnet(dummy_input)
                    last_warmup_time = time.time()
                except:
                    pass

    def terminate(self):
        self.stop_threads = True