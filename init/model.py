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
        初始化 Detector 實例並啟動背景執行緒。

        Parameters:
        frame_num (int): 攝影機編號(0=進入, 1=離開)
        system (object): 全域系統狀態物件，包含 .state, .mtcnn, .model_clothes
        """
        self.system = system
        self.frame_num = frame_num
        self.TIMEZONE = pytz.timezone('Asia/Taipei')
        self.stop_threads = False
        self.last_face_time = 0       # 最後一次偵測到人臉的時間
        self.last_no_face_log_time = 0 # 用於控制 "未偵測到人臉" Log 的輸出頻率
        self.clothe_time = [0, 0, 0]  # 各項穿著檢測的最後更新時間
        threading.Thread(target=self.face_detector, daemon=True).start()

    def face_detector(self):
        """
        背景執行的臉部與衣著偵測流程：
        - 透過 DETECTION_INTERVAL 控制偵測頻率，降低 CPU 負載。
        - 移除第二次偵測，確保處理時間穩定。
        - 若啟用衣著辨識且為主畫面，則觸發衣著辨識。
        - 若超過一定時間未偵測到人臉，則清除衣著標記。
        - 每 10 秒進行模型暖機以減少推論延遲。
        """
        last_box = None
        last_points = None
        last_time = 0
        last_detection_time = 0
        DETECTION_INTERVAL = self.system.state.detection_interval
        dummy_input = tensor_test_img[0]

        while not self.stop_threads:
            now = time.time()
            # 如果有新的畫面可以處理
            if self.system.state.frame[self.frame_num] is not None:
                
                if now - last_detection_time > DETECTION_INTERVAL:
                    last_detection_time = now # 重置計時器
                    
                    self.system.state.max_box[self.frame_num] = last_box
                    self.system.state.max_points[self.frame_num] = last_points
                    new_frame = self.system.state.frame[self.frame_num].copy()
                    
                    # Capture high-res frame snapshot
                    new_high_res = None
                    if self.system.state.frame_high_res is not None and self.system.state.frame_high_res[self.frame_num] is not None:
                         new_high_res = self.system.state.frame_high_res[self.frame_num].copy()

                    # 預設為無臉框
                    box = None
                    points = None

                    # 1. 偵測來源直接使用原生解析度影像 (GlobalState.frame 現已儲存原生影像)
                    detect_source = new_frame
                    h_source, w_source = detect_source.shape[:2]

                    # 2. 計算 ROI (用於過濾與衣著遮罩)
                    close_N = 6
                    if CONFIG[CAMERA[self.frame_num]]["close"]:
                        close_N = 8
                    roi_x1 = w_source // close_N
                    roi_x2 = (close_N - 1) * w_source // close_N

                    # 3. 執行偵測 (使用全畫面偵測以確保大臉能被捕捉)
                    detect_start_time = time.monotonic()
                    
                    # Full Frame Detection
                    boxes, _, landmarks = self.system.mtcnn.detect(detect_source, landmarks=True)
                    
                    detect_duration = time.monotonic() - detect_start_time
                    
                    # Add structured performance logging
                    perf_data = {
                        "type": "detection",
                        "duration_sec": detect_duration,
                        "camera_name": CAM_NAME_MAP.get(self.frame_num, f"Cam {self.frame_num}"),
                        "timestamp": datetime.now(self.TIMEZONE).isoformat(),
                        "resolution": f"{w_source}x{h_source}",
                        "strategy": "FullFrame_Native"
                    }
                    PERF_LOGGER.info(json.dumps(perf_data))

                    if boxes is not None:
                        self.last_face_time = time.time()
                        x1, y1, x2, y2 = map(int, boxes[0])
                        points = landmarks[0].copy()
                        
                        # ROI 過濾 (檢查中心點是否在 ROI 內)
                        center_x = (x1 + x2) / 2
                        if center_x < roi_x1 or center_x > roi_x2:
                            # 雖偵測到人臉但不在關注區域內 -> 忽略
                            box = None
                        else:
                            # [2026-01-08 修正] 距離過濾 (Distance Filter)
                            # 若人臉寬度小於 min_face 的 70%，視為過遠，不顯示框也不辨識
                            # 避免使用者誤以為系統正在辨識卻失敗
                            face_width = x2 - x1
                            min_face_val = self.system.state.min_face[self.frame_num]
                            
                            if face_width < (min_face_val * POTENTIAL_MISS_RATIO):
                                box = None
                            else:
                                # 人臉有效 (無需縮放，座標即為原生座標)
                                box = [x1, y1, x2, y2]

                                # 若為主要畫面（frame_num=0）且開啟衣著檢測，則執行衣著辨識
                            if CONFIG["Clothes_show"] and self.frame_num == 0 and \
                               (not self.system.state.clothes[0] or not self.system.state.clothes[2]):
                                # 準備 mask_frame 給衣著偵測 (裁切 ROI)
                                self.mask_frame = detect_source[:, roi_x1:roi_x2].copy()
                                # 衣著偵測使用的是 Mask 內的相對座標，故傳入 ROI 起點作為 Offset
                                self.clothes_detector(roi_x1)

                    else:
                        # 若超過 1 秒沒偵測到臉，則檢查是否重置衣著狀態
                        if time.time() - self.last_face_time > 1 and \
                                CONFIG["Clothes_show"] and self.frame_num == 0:
                            for i in range(3):
                                if time.time() - self.clothe_time[i] > 3:
                                    self.system.state.clothes[i] = False

                    # 更新 MTCNN 結果與最新臉框
                    self.system.state.max_box[self.frame_num] = box
                    self.system.state.max_points[self.frame_num] = points
                    self.system.state.frame_mtcnn[self.frame_num] = new_frame
                    self.system.state.frame_mtcnn_high_res[self.frame_num] = new_high_res
                    last_box = box
                    last_points = points
                    last_time = time.time()

            # 每 10 秒暖機一次 MTCNN 模型，避免延遲推論
            elif time.time() - last_time > 10 and self.frame_num == 0:
                try:
                    if CONFIG["Clothes_show"]:
                        _ = self.system.model_clothes(
                            source=dummy_input.unsqueeze(0),
                            iou=0.45,
                            conf=0.2,
                            verbose=False
                        )[0]
                    __, _, _ = self.system.mtcnn.detect(
                        dummy_input, landmarks=True)
                    last_time = time.time()
                except:
                    pass

            # **優化**: 使用 0.01 秒休眠，避免忙碌等待並讓出 CPU
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
            today_str = datetime.now(self.TIMEZONE).strftime('%Y_%m_%d')
            time_str = datetime.now(self.TIMEZONE).strftime('%H;%M;%S')
            
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
        msg (str): 詳細的評估訊息 (包含數值與門檻)
        """
        # MTCNN points: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = points[0]
        right_eye = points[1]
        nose = points[2]
        left_mouth = points[3]
        right_mouth = points[4]

        # ---------------------------------------------------------
        # 1. 畫面置中檢查 (Center Alignment)
        # ---------------------------------------------------------
        face_center_x = (box[0] + box[2]) / 2
        frame_center_x = frame_w / 2
        offset = abs(face_center_x - frame_center_x)
        limit_offset = frame_w * 0.15 # 允許偏離 15%
        
        if offset > limit_offset:
            return 0.0, f"未置中 (偏離 {offset:.1f}px > 容許 {limit_offset:.1f}px)"

        # ---------------------------------------------------------
        # 2. 特徵點完整性檢查 (Visibility)
        # ---------------------------------------------------------
        margin = 5
        for i, p in enumerate(points):
            if p[0] < margin or p[0] > frame_w - margin or \
               p[1] < margin or p[1] > frame_h - margin:
                 return 0.0, f"特徵點被切除/遮擋 (點{i}座標 {p} 超出邊界)"

        # ---------------------------------------------------------
        # 3. 垂直比例檢查 (Pitch)
        # ---------------------------------------------------------
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2
        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2
        nose_y = nose[1]
        
        h_upper = nose_y - eye_mid_y # 鼻眼距
        h_lower = mouth_mid_y - nose_y # 鼻嘴距
        
        if h_upper <= 0 or h_lower <= 0:
             return 0.0, "特徵點垂直錯位 (無法計算比例)"

        v_ratio = h_lower / h_upper
        
        # 3.1 抬頭檢查 (優先排除，避免透視導致嘴寬誤判)
        # 放寬門檻: 從 1.6 調升至 2.1，適應極端臉型或廣角鏡頭畸變
        if v_ratio > 2.1:
             return 0.0, f"抬頭 (垂直比例 {v_ratio:.2f} > 門檻 2.10)"

        # 3.2 低頭檢查 (涵蓋所有 V-Ratio 偏低情況)
        # 放寬門檻: 從 0.70 降至 0.60
        if v_ratio < 0.60:
             return 0.0, f"低頭 (垂直比例 {v_ratio:.2f} < 門檻 0.60)"

        # ---------------------------------------------------------
        # 4. 側臉/未正視檢查 (Yaw & Gaze)
        # ---------------------------------------------------------
        dist_l_eye = abs(nose[0] - left_eye[0])
        dist_r_eye = abs(right_eye[0] - nose[0])
        
        # 4.1 絕對距離檢查 (眼鼻距)
        face_w = max(10, box[2] - box[0])
        min_dist_ratio = min(dist_l_eye, dist_r_eye) / face_w
        
        # 放寬門檻: 從 0.15 降至 0.12，進一步減少誤報
        if min_dist_ratio < 0.12:
            eye_side = "左眼" if dist_l_eye < dist_r_eye else "右眼"
            return 0.0, f"未正視鏡頭/極端側臉 ({eye_side}鼻距比 {min_dist_ratio:.2f} < 門檻 0.12)"

        # 4.2 左右對稱性檢查
        if dist_l_eye > 0 and dist_r_eye > 0:
            ratio_yaw = min(dist_l_eye, dist_r_eye) / max(dist_l_eye, dist_r_eye)
            # 放寬門檻: 從 0.65 降至 0.55
            if ratio_yaw < 0.55:
                return 0.0, f"側臉 (左右對稱比 {ratio_yaw:.2f} < 門檻 0.55)"

        # 5.3 歪頭檢查 (Roll)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        
        if angle > 15:
             return 0.0, f"歪頭 (角度 {angle:.1f} > 門檻 15.0)"

        # DEBUG: 輸出通過檢查的詳細數值，方便調校
        # LOGGER.debug(f"品質檢查通過: V-Ratio={v_ratio:.2f}, NM-Dist={h_lower:.1f}(Ref:{eye_dist*0.5:.1f}), Yaw-Ratio={ratio_yaw:.2f}")
        return 1.0, f"Pass (V:{v_ratio:.2f}, NM:{h_lower:.1f})"

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
        Z_SCORE_THRESHOLD = 1.2 # Z-Score 門檻，最高分需顯著高於平均值 (2.5 通常代表 99% 信心水準)

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

            # 檢查人臉品質
            # 傳入 box, points 與畫面寬高
            quality_score, quality_msg = self.check_face_quality(_box, _points, frame_w, frame_h)
            
            # 若品質不達標 (Score=0)，記錄原因並跳過
            if quality_score == 0.0:
                 # 這裡強制 Log 過濾原因
                 LOGGER.info(f"[{camera_name}][品質過濾] {quality_msg}")
                 
                 # 根據原因給出明確的 UI 提示 (Actionable Hint)
                 if "未置中" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請站到中間"
                     self.system.speaker.say("請站到中間", "hint_center", priority=2)
                 elif "特徵點被切除/遮擋" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "臉部被遮擋/切到"
                     self.system.speaker.say("臉部被遮擋或切到", "hint_occlusion", priority=2)
                 elif "低頭" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請抬頭"
                     self.system.speaker.say("請抬頭", "hint_look_up", priority=2)
                 elif "抬頭" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請低頭"
                     self.system.speaker.say("請低頭", "hint_look_down", priority=2)
                 elif "未正視" in quality_msg or "側臉" in quality_msg or "歪頭" in quality_msg:
                     self.system.state.hint_text[self.frame_num] = "請正視鏡頭"
                     self.system.speaker.say("請正視鏡頭", "hint_look_straight", priority=2)
                 else:
                     self.system.state.hint_text[self.frame_num] = "調整位置"
                     self.system.speaker.say("請調整位置", "hint_adjust", priority=2)
                     
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
                f"信賴度: {confidence:.2%}{quality_info}, Z-Score: {z_score:.2f} [評級: {rating_str}]"
            )
            
            # 策略：只要是被品質扣分導致失敗的（原本分數夠高，但扣完變低），或者成功但有被扣分的，都強制 Log 出來方便除錯
            is_penalized_failure = (raw_confidence >= self.CONFIDENCE_THRESHOLD) and (confidence < self.CONFIDENCE_THRESHOLD)
            
            # 根據評級決定是否為有效辨識
            if is_reliable:
                LOGGER.info(log_message)
                person_id = predicted_id

                # 觸發成功事件 (例如：開門)
                self.system.state.same_people[self.frame_num] = confidence
                self.system.state.same_zscore[self.frame_num] = z_score # Sync Z-Score to GlobalState
                # 更新UI顯示的人員名稱
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
