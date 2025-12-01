import json
import os
import threading
import time

import cv2
import numpy as np
import numba as nb
import torch
from init.log import LOGGER
from datetime import datetime
import pytz
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

#載入設定檔
with open(os.path.join(os.path.dirname(__file__), "../config.json"), "r", encoding="utf-8") as json_file:
    CONFIG = json.load(json_file)
CAMERA = {0:"inCamera", 1:"outCamera"}

test_img = cv2.imread(os.path.join(os.path.dirname(__file__), "../other/test_img.jpg"))
test_img = cv2.resize(test_img,(224,224))
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
        self.stop_threads = False
        self.last_face_time = 0       # 最後一次偵測到人臉的時間
        self.clothe_time = [0, 0, 0]  # 各項穿著檢測的最後更新時間
        threading.Thread(target=self.face_detector, daemon=True).start()

    def face_detector(self):
        """
        背景執行的臉部與衣著偵測流程：
        - 每次處理新影像進行臉部偵測。
        - 若啟用衣著辨識且為主畫面，則觸發衣著辨識。
        - 若超過一定時間未偵測到人臉，則清除衣著標記。
        - 每 10 秒進行模型暖機以減少推論延遲。
        """
        last_box = None
        last_points = None
        last_time = 0
        dummy_input = tensor_test_img[0]#torch.zeros(3, 224, 224)  # 用於 MTCNN 模型暖機的假圖像

        while not self.stop_threads:
            # 如果有新的畫面可以處理
            if self.system.state.frame[self.frame_num] is not None:
                self.system.state.max_box[self.frame_num] = last_box
                self.system.state.max_points[self.frame_num] = last_points
                new_frame = self.system.state.frame[self.frame_num].copy()

                # 預設為無臉框
                box = None
                points = None

                # 套用遮罩處理取得 ROI 區域
                mask_frame, X_offset = self.apply_mask(new_frame)
                self.mask_frame = mask_frame.copy()

                # 嘗試偵測臉部
                boxes, _, landmarks = self.system.mtcnn.detect(mask_frame, landmarks=True)
                if boxes is None:
                    # 若失敗，使用直方圖均衡化後重試
                    boxes, _, landmarks = self.system.mtcnn.detect(self.equalize(mask_frame), landmarks=True)

                if boxes is not None:
                    self.last_face_time = time.time()
                    x1, y1, x2, y2 = map(int, boxes[0])
                    x1, x2 = x1 + X_offset, x2 + X_offset
                    box = [x1, y1, x2, y2]
                    points = landmarks[0]

                    # 若為主要畫面（frame_num=0）且開啟衣著檢測，則執行衣著辨識
                    if CONFIG["Clothes_show"] and self.frame_num == 0 and \
                       (not self.system.state.clothes[0] or not self.system.state.clothes[2]):
                        self.clothes_detector(X_offset)

                # 若超過 1 秒沒偵測到臉，則檢查是否重置衣著狀態
                elif time.time() - self.last_face_time > 1 and \
                     CONFIG["Clothes_show"] and self.frame_num == 0:
                    for i in range(3):
                        if time.time() - self.clothe_time[i] > 3:
                            self.system.state.clothes[i] = False

                # 更新 MTCNN 結果與最新臉框
                self.system.state.max_box[self.frame_num] = box
                self.system.state.max_points[self.frame_num] = points
                self.system.state.frame_mtcnn[self.frame_num] = new_frame
                last_box = box
                last_points = points
                last_time = time.time()

            # 每 10 秒暖機一次 MTCNN 模型，避免延遲推論
            elif time.time() - last_time > 10 and self.frame_num == 0:
                try:
                    if CONFIG["Clothes_show"] :
                        _ = self.system.model_clothes(
                            source=dummy_input.unsqueeze(0),
                            iou=0.45,
                            conf=0.2,
                            verbose=False
                        )[0]
                    __, _, _ = self.system.mtcnn.detect(dummy_input, landmarks=True)
                    last_time = time.time()
                except:
                    pass

            time.sleep(0.00001)

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
        close_N = 3
        if CONFIG[CAMERA[self.frame_num]]["close"]:
            close_N = 4

        # 產生白色矩形遮罩
        cv2.rectangle(
            mask,
            (width // close_N, 0),
            ((close_N - 1) * width // close_N, height),
            (255, 255, 255), -1
        )

        # 套用遮罩後回傳遮罩區域與偏移量
        masked_frame = cv2.bitwise_and(frame, mask)
        masked_frame = frame[0:, width // close_N : (close_N - 1) * width // close_N]
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
        self.CONFIDENCE_THRESHOLD = 0.65     # 單次辨識的信賴度門檻
        self.RECOGNITION_COOLDOWN = 5        # 同一個攝影機在辨識成功後的冷卻時間(秒)

        self.TIMEZONE = pytz.timezone('Asia/Taipei')

        threading.Thread(target=self.face_comparison, daemon=True).start()

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
        """
        last_warmup_time = 0
        dummy_input = tensor_test_img

        while not self.stop_threads:
            time.sleep(0.01)
            now = time.time()

            # 清除畫面上的人員名稱（如果超過顯示時間）
            if self.display_state['person_id'] != 'None' and \
               now - self.display_state['last_update'] > self.DISPLAY_STATE_HOLD_SECONDS:
                self._update_display_state('None')

            # 清空歷史紀錄，確保不顯示信賴度分數
            self.system.state.display_history[self.frame_num] = []

            # 檢查是否有臉部框和特徵點
            _box = self.system.state.max_box[self.frame_num]
            _points = self.system.state.max_points[self.frame_num]
            if _box is None or _points is None or self.system.state.frame_mtcnn[self.frame_num] is None:
                continue

            # 檢查臉部大小是否足夠
            if _box[2] - _box[0] < self.system.state.min_face:
                continue
            
            # 檢查是否在辨識冷卻時間內
            if now - self.last_recognition_time < self.RECOGNITION_COOLDOWN:
                continue

            # 提取人臉特徵向量
            try:
                # Convert BGR frame to PIL Image
                frame_image = Image.fromarray(cv2.cvtColor(self.system.state.frame_mtcnn[self.frame_num], cv2.COLOR_BGR2RGB))
                
                # Crop face without forehead using shared function
                img_cropped = crop_face_without_forehead(frame_image, _box, _points)
                
                # Add batch dimension and get embedding
                face_embedding_list = self.system.resnet(img_cropped.unsqueeze(0))
                
                if face_embedding_list is None or len(face_embedding_list) == 0:
                    continue
                current_face_vec = face_embedding_list[0].detach().numpy()
            except Exception as e:
                LOGGER.error(f"[ERROR][Cam {self.frame_num}] 臉部特徵提取失敗: {e}")
                continue

            # 進行預測與信賴度計算
            try:
                predicted_id = self.system.svc.predict([current_face_vec])[0]
                registered_embeddings = self.system.state.features_dict.get(predicted_id, [])
                if not registered_embeddings:
                    continue
                total_similarity = sum(cosine_similarity(current_face_vec, emb) for emb in registered_embeddings)
                confidence = total_similarity / len(registered_embeddings)
            except Exception as e:
                LOGGER.error(f"[ERROR][Cam {self.frame_num}] 預測或信賴度計算失敗: {e}")
                continue

            # 標記每一次辨識事件（無論成功與否）
            log_time = datetime.now(self.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
            staff_name = self.system.state.features_dict.get("id_name", {}).get(predicted_id, "未知")
            LOGGER.info(f"{log_time} [辨識事件][Cam {self.frame_num}] 偵測到 {staff_name} (ID: {predicted_id}), 信賴度: {confidence:.2%}")

            # 如果單次信賴度超過門檻，則直接視為成功
            if confidence >= self.CONFIDENCE_THRESHOLD:
                person_id = predicted_id
                
                # 觸發成功事件 (例如：開門)
                self.system.state.same_people[self.frame_num] = confidence # Store confidence here
                # 更新UI顯示的人員名稱
                self._update_display_state(person_id)
                # 更新最後辨識成功的時間
                self.last_recognition_time = now

            # 模型暖機
            if time.time() - last_warmup_time > 10 and self.frame_num == 0:
                try:
                    _ = self.system.resnet(dummy_input)
                    last_warmup_time = time.time()
                except:
                    pass

    def terminate(self):
        self.stop_threads = True
