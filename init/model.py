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
                                # 存入系統以便 Comparison 呼叫 check_gaze
                                self.system.mp_detectors[self.frame_num] = self.mp_handler

                    self.system.state.max_box[self.frame_num] = box
                    self.system.state.max_points[self.frame_num] = points
                    self.system.state.frame_mtcnn[self.frame_num] = new_frame
                    self.system.state.frame_mtcnn_high_res[self.frame_num] = new_high_res
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

        self.DISPLAY_STATE_HOLD_SECONDS = 3  # 辨識成功後，名稱顯示的持續時間
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

    def check_face_quality(self, box, points, frame_w, frame_h):
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
        # [2026-01-11] 全面移交給 MediaPipeHandler (PnP + Gaze)
        face_w = max(10, box[2] - box[0])
        if face_w > 100:
            mp_handler = self.system.mp_detectors.get(self.frame_num)
            if mp_handler:
                is_looking, gaze_msg = mp_handler.check_gaze(0)
                if not is_looking:
                    # 直接回傳具體的錯誤訊息 (例如: "Head: 歪頭", "Eye V: Down")
                    return 0.0, f"{gaze_msg}"

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

            # Check if we have a detected face
            _box = self.system.state.max_box[self.frame_num]
            _points = self.system.state.max_points[self.frame_num]
            if _box is None or _points is None or self.system.state.frame_mtcnn[self.frame_num] is None:
                continue
            
            # 取得畫面尺寸 (用於置中與邊界檢查)
            frame_curr = self.system.state.frame_mtcnn[self.frame_num]
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
            # 這能有效阻擋尺寸忽大忽小的路燈光暈，避免誤觸 "請靠近" 或 "請抬頭"
            current_hour = datetime.now(self.TIMEZONE).hour
            is_night_mode = (current_hour >= 18 or current_hour < 6)
            
            # 提取特徵向量 (為了夜間檢查或後續辨識)
            # 為了避免重複提取，這裡先統一做一次
            current_face_vec = None
            try:
                frame_to_use = self.system.state.frame_mtcnn[self.frame_num]
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
                if self.system.state.ann_index and self.system.state.ann_index.index.ntotal > 0:
                    dists, _ = self.system.state.ann_index.search(current_face_vec, k=1)
                    if dists[0] < 0.4:
                        # 相似度太低，認定為雜訊 (路燈)，完全忽略
                        # LOGGER.debug(f"[{camera_name}][夜間全域過濾] 忽略雜訊 (寬度: {face_width}, 相似度: {dists[0]:.2f})")
                        continue

            if face_width < min_face_threshold:
                # [2026-01-10] 修正：距離不夠時，立刻清除原本的人員顯示，防止誤導
                # 不管是不是 "Potential Miss"，只要進到這個「過小」的分支，就不應顯示之前的人員名字
                if self.display_state['person_id'] != 'None':
                    self._update_display_state('None')

                # --- 新增: 潛在辨識失敗偵測 (Near Miss Detection) ---
                # 動態調整比率：夜間(18:00-06:00)設為 0.9 以過濾路燈；日間維持 0.8
                current_potential_ratio = 0.9 if is_night_mode else self.potential_miss_ratio
                potential_threshold = min_face_threshold * current_potential_ratio
                
                # 如果寬度落在 [min_face * ratio, min_face) 區間，視為有意圖但失敗
                if face_width >= potential_threshold:
                    # 限流: 同一鏡頭 3 秒內只記錄一次，避免洗版
                    if now - self.last_potential_miss_log_time > 3:
                        
                        # 使用當前原生畫面截圖
                        snapshot = self.system.state.frame_mtcnn[self.frame_num]
                            
                        saved_path = "無影像"
                        if snapshot is not None:
                            saved_path = self._save_potential_miss_image(snapshot, face_width, min_face_threshold, camera_name)
                            
                        LOGGER.info(f"[{camera_name}][潛在失敗] 偵測到人臉但過小 (寬度: {face_width}, 門檻: {min_face_threshold}) - 已存檔: {saved_path}")
                        self.last_potential_miss_log_time = now
                        
                        # 設定 UI 提示
                        self.system.state.hint_text[self.frame_num] = "請靠近鏡頭"
                        self.hint_clear_time = now + 2.0

                        # --- 新增: 語音提示 (若有簽到語音正在播或排隊則自動放棄) ---
                        # Priority=2 (Normal)，若忙碌會自動丟棄，無需手動計時 CD 或簽到冷卻
                        self.system.speaker.say("請靠近鏡頭", "hint_closer", priority=2)
                        self.last_hint_speak_time = now
                
                continue

            # [2026-01-10] 修正：臉部過大時，因廣角畸變嚴重導致辨識失效
            # 改為引導使用者後退，且此檢查優先於任何品質評估或特徵提取
            if face_width >= CONFIG["max_face"]:
                # LOGGER.info(f"[{camera_name}] 人臉過大 ({face_width}px >= {CONFIG['max_face']}px)，提示使用者後退。")
                self.system.state.hint_text[self.frame_num] = "請稍微後退"
                self.system.speaker.say("請稍微後退", "hint_move_back", priority=2)
                self.hint_clear_time = time.time() + 1.5
                
                if self.display_state['person_id'] != 'None':
                    self._update_display_state('None')
                continue

            # 檢查人臉品質
            # 傳入 box, points 與畫面寬高
            quality_score, quality_msg = self.check_face_quality(_box, _points, frame_w, frame_h)
            
            # 若品質不達標 (Score=0)，記錄原因並跳過
            if quality_score == 0.0:
                 # 這裡強制 Log 過濾原因
                 LOGGER.info(f"[{camera_name}][品質過濾] {quality_msg}")
                 
                 # 根據原因給出明確的 UI 提示 (Actionable Hint)
                 if "低頭" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請抬頭"
                     self.system.speaker.say("請抬頭", "hint_look_up", priority=2)
                 elif "抬頭" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請低頭"
                     self.system.speaker.say("請低頭", "hint_look_down", priority=2)
                 elif "未置中" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請站到中間"
                     self.system.speaker.say("請站到中間", "hint_center", priority=2)
                 elif "斜視" in quality_msg or "未正視" in quality_msg or "側臉" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請正視鏡頭"
                     self.system.speaker.say("請正視鏡頭", "hint_look_straight", priority=2)
                 else:
                     self.system.state.hint_text[self.frame_num] = "請對準鏡頭"
                     self.system.speaker.say("請對準鏡頭", "hint_occlusion", priority=2)
                     
                 self.hint_clear_time = now + 1.0
                 continue

            # 提取人臉特徵向量 (若前面尚未提取成功)
            if current_face_vec is None:
                try:
                    comparison_start_time = time.monotonic()
                    
                    frame_to_use = self.system.state.frame_mtcnn[self.frame_num]
                    box_to_use = list(_box)
                    points_to_use = _points.copy()
                    
                    frame_image = Image.fromarray(cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2RGB))
                    img_cropped = crop_face_without_forehead(frame_image, box_to_use, points_to_use)
                    face_embedding_list = self.system.resnet(img_cropped.unsqueeze(0))

                    if face_embedding_list is None or len(face_embedding_list) == 0:
                        LOGGER.warning(f"[{camera_name}] 特徵提取回傳空值 (ResNet Output is None)")
                        continue
                    current_face_vec = face_embedding_list[0].detach().numpy()
                except Exception as e:
                    LOGGER.error(f"[ERROR][{camera_name}] 臉部特徵提取失敗: {e}")
                    continue
            else:
                 comparison_start_time = time.monotonic() # 若已提取，只需重設計時

            # 進行預測與信賴度計算 (Z-Score 離群值分析)
            try:
                # 檢查 AnnIndex 是否準備就緒
                if self.system.state.ann_index is None or self.system.state.ann_index.index is None or self.system.state.ann_index.index.ntotal == 0:
                    LOGGER.warning(f"[{camera_name}] Faiss 索引未準備就緒或為空，跳過比對。")
                    predicted_id = "None"
                    confidence = 0.0
                    z_score = 0.0
                    top_k_similarities = np.array([])
                    comparison_duration = time.monotonic() - comparison_start_time
                    num_people_in_index = 0
                else:
                    # 使用 Faiss 進行搜尋 (k=5 獲取前5個最相似結果)
                    # Faiss returns Inner Product (cosine similarity for normalized vectors)
                    distances, faiss_person_ids = self.system.state.ann_index.search(current_face_vec, k=min(5, self.system.state.ann_index.index.ntotal))

                    if faiss_person_ids is None or len(faiss_person_ids) == 0:
                        predicted_id = "None"
                        confidence = 0.0
                        z_score = 0.0
                        top_k_similarities = np.array([])
                    else:
                        # Store all top-k similarities for Z-score calculation
                        top_k_similarities = np.array(distances)
                        
                        best_match_id = faiss_person_ids[0]
                        predicted_id = best_match_id
                        max_similarity = distances[0] # The highest cosine similarity
                        raw_confidence = max_similarity
                        
                        # 套用品質懲罰
                        confidence = raw_confidence * quality_score

                        # Z-Score 離群值分析 (使用 top-k 結果)
                        z_score = 0.0
                        if len(top_k_similarities) > 1: # 至少需要兩個結果來計算標準差
                            mean_score = np.mean(top_k_similarities)
                            std_dev_score = np.std(top_k_similarities)

                            if std_dev_score > 0:
                                z_score = (raw_confidence - mean_score) / std_dev_score # Z-Score 用原始分數算較準
                            else: # 所有分數都相同，若分數高且達門檻，視為通過
                                z_score = Z_SCORE_THRESHOLD if confidence >= self.CONFIDENCE_THRESHOLD else 0
                        else: # 只有一個結果時，無法計算 Z-score，預設為通過（若信賴度達標）
                            z_score = Z_SCORE_THRESHOLD if confidence >= self.CONFIDENCE_THRESHOLD else 0
                    
                    comparison_duration = time.monotonic() - comparison_start_time
                    num_people_in_index = self.system.state.ann_index.index.ntotal if self.system.state.ann_index.index else 0
                
                # Add structured performance logging
                perf_data = {
                    "type": "comparison",
                    "duration_sec": comparison_duration,
                    "person_id": predicted_id,
                    "num_compared": num_people_in_index,
                    "camera_name": camera_name,
                    "timestamp": datetime.now(self.TIMEZONE).isoformat()
                }
                PERF_LOGGER.info(json.dumps(perf_data))

            except Exception as e:
                LOGGER.error(f"[ERROR][{camera_name}] 預測或信賴度計算失敗: {e}")
                continue

            # 標記每一次辨識事件（無論成功與否）
            log_time = datetime.now(
                self.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
            staff_name = self.system.state.features_dict.get(
                "id_name", {}).get(predicted_id, "未知")
            
            # 新增：決定辨識品質評級
            is_reliable = confidence >= self.CONFIDENCE_THRESHOLD and z_score >= Z_SCORE_THRESHOLD
            is_visitor = confidence < self.VISITOR_CONF_THRESHOLD
            
            rating_str = "模糊 (Ambiguous)"
            if is_reliable:
                rating_str = "可靠 (Reliable)"
            elif is_visitor:
                rating_str = "訪客 (Visitor)"
            
            # 格式化品質資訊 (若有懲罰才顯示細節)
            quality_info = ""
            if quality_score < 1.0:
                quality_info = f" (原:{raw_confidence:.2%}, 品質:{quality_score:.2f} {quality_msg})"

            # 更新日誌
            log_message = (
                f"{log_time} [辨識事件][{camera_name}] 偵測到 {staff_name} (ID: {predicted_id}), "
                f"信賴度: {confidence:.2%}{quality_info}, Z-Score: {z_score:.2f}, Width: {face_width} px [評級: {rating_str}]"
            )
            
            # 策略：只要是被品質扣分導致失敗的（原本分數夠高，但扣完變低），或者成功但有被扣分的，都強制 Log 出來方便除錯
            is_penalized_failure = (raw_confidence >= self.CONFIDENCE_THRESHOLD) and (confidence < self.CONFIDENCE_THRESHOLD)
            
            # 根據評級決定是否為有效辨識
            if is_reliable:
                LOGGER.info(log_message)
                person_id = predicted_id

                # 鎖定辨識成功當下的影像快照 (直接使用 BGR，無需轉換)
                try:
                    self.system.state.success_frame[self.frame_num] = frame_to_use.copy()
                except Exception as e:
                    LOGGER.error(f"儲存成功快照失敗: {e}")

                # 觸發成功事件 (例如：開門)
                # [2026-01-09] 針對個人的語音/API冷卻機制 (播完後2秒)
                # 使用 speaker 的狀態來判斷是否正在播或剛播完
                speaker = self.system.speaker
                is_cooldown = False
                cooldown_reason = ""
                
                with speaker.status_lock:
                    last_start = speaker.last_start_time.get(person_id, 0)
                    last_end = speaker.last_end_time.get(person_id, 0)
                
                # 判斷 1: 是否正在播放 (Start > End)
                if last_start > last_end:
                    # [安全機制] 若播放狀態持續超過 10 秒，視為異常(卡死)，強制解除冷卻
                    if now - last_start > 10.0:
                         is_cooldown = False
                         LOGGER.warning(f"[{camera_name}] 偵測到語音狀態卡死 (>10s)，強制解除人員 {staff_name} 的冷卻鎖。")
                    else:
                        is_cooldown = True
                        cooldown_reason = "正在播放中"
                # 判斷 2: 是否剛播完 (Now - End < 2.0)
                elif now - last_end < 2.0:
                    is_cooldown = True
                    cooldown_reason = f"剛播完 (剩餘 {2.0 - (now - last_end):.1f}s)"
                
                if not is_cooldown:
                    self.system.state.same_people[self.frame_num] = confidence
                    self.system.state.same_zscore[self.frame_num] = z_score
                    self.system.state.same_width[self.frame_num] = face_width # 記錄臉寬
                    # 注意：這裡不更新 last_api_trigger_time，因為那是舊邏輯
                    # 新邏輯的時間更新會在 speaker.say 被呼叫時由 speaker 內部處理
                else:
                    LOGGER.debug(f"[{camera_name}] 人員 {staff_name} 冷卻中 ({cooldown_reason})，跳過語音觸發。")

                # 更新UI顯示的人員名稱 (不受冷卻影響，確保持續顯示)
                self._update_display_state(person_id)
                # 更新最後辨識成功的時間
                self.last_recognition_time = now
            elif is_penalized_failure:
                 # 這是關鍵：原本可以過，但因為側臉被擋下來的案例 -> 強制 Log
                 LOGGER.info(f"[側臉攔截] {log_message}")
            elif is_visitor:
                 LOGGER.info(log_message)
                 # 標記為訪客，但不觸發打卡
                 self.system.state.same_people[self.frame_num] = 0.0
                 self._update_display_state("__VISITOR__")
            else: # Ambiguous Case (非扣分導致的普通模糊)
                 # LOGGER.info(log_message) # 可選：是否要全開 Log
                 pass

            # 模型暖機
            if time.time() - last_warmup_time > 10 and self.frame_num == 0:
                try:
                    _ = self.system.resnet(dummy_input)
                    last_warmup_time = time.time()
                except:
                    pass

    def terminate(self):
        self.stop_threads = True
