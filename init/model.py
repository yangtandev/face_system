
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
        last_time = 0
        dummy_input = tensor_test_img[0]#torch.zeros(3, 224, 224)  # 用於 MTCNN 模型暖機的假圖像

        while not self.stop_threads:
            # 如果有新的畫面可以處理
            if self.system.state.frame[self.frame_num] is not None:
                self.system.state.max_box[self.frame_num] = last_box
                new_frame = self.system.state.frame[self.frame_num].copy()

                # 預設為無臉框
                box = None

                # 套用遮罩處理取得 ROI 區域
                mask_frame, X_offset = self.apply_mask(new_frame)
                self.mask_frame = mask_frame.copy()

                # 嘗試偵測臉部
                boxes, _ = self.system.mtcnn.detect(mask_frame)
                if boxes is None:
                    # 若失敗，使用直方圖均衡化後重試
                    boxes, _ = self.system.mtcnn.detect(self.equalize(mask_frame))

                if boxes is not None:
                    self.last_face_time = time.time()
                    x1, y1, x2, y2 = map(int, boxes[0])
                    x1, x2 = x1 + X_offset, x2 + X_offset
                    box = [x1, y1, x2, y2]

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
                self.system.state.frame_mtcnn[self.frame_num] = new_frame
                last_box = box
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
                    __, _ = self.system.mtcnn.detect(dummy_input)
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
    - 採用帶時間衰減的信心分數累加器，提升辨識穩定性。
    - 引入顯示狀態保持機制，解決畫面閃爍問題。
    - 提供詳細的日誌記錄，包含分數組成歷史。
    """
    def __init__(self, frame_num, system):
        self.system = system
        self.frame_num = frame_num
        self.stop_threads = False

        self.recognition_states = {}
        self.display_state = {'person_id': 'None', 'last_update': 0}

        self.DISPLAY_STATE_HOLD_SECONDS = 3
        self.CONFIDENCE_THRESHOLD = 0.65
        self.SUCCESS_SCORE_THRESHOLD = 2.0
        self.STATE_EXPIRATION_SECONDS = 5

        self.TIMEZONE = pytz.timezone('Asia/Taipei')

        threading.Thread(target=self.face_comparison, daemon=True).start()

    def _cleanup_expired_states(self):
        now = time.time()
        expired_keys = [
            person_id for person_id, state in self.recognition_states.items()
            if now - state['last_seen'] > self.STATE_EXPIRATION_SECONDS
        ]
        if expired_keys:
            for key in expired_keys:
                del self.recognition_states[key]
            if not self.recognition_states:
                self.system.state.display_history[self.frame_num] = []

    def _update_display_state(self, person_id):
        self.display_state['person_id'] = person_id
        self.display_state['last_update'] = time.time()
        self.system.state.same_class[self.frame_num] = person_id

    def face_comparison(self):
        last_warmup_time = 0
        dummy_input = tensor_test_img

        while not self.stop_threads:
            time.sleep(0.01)
            now = time.time()
            self._cleanup_expired_states()

            if self.display_state['person_id'] != 'None' and \
               now - self.display_state['last_update'] > self.DISPLAY_STATE_HOLD_SECONDS:
                if not self.recognition_states:
                    self._update_display_state('None')
                    self.system.state.display_history[self.frame_num] = []

            self.system.state.same_people[self.frame_num] = 0

            if self.system.state.max_box[self.frame_num] is None or \
               self.system.state.frame_mtcnn[self.frame_num] is None:
                continue

            _box = self.system.state.max_box[self.frame_num]
            if _box[2] - _box[0] < self.system.state.min_face:
                continue

            try:
                face_embedding_list = self.system.resnet(self.system.mtcnn.extract(self.system.state.frame_mtcnn[self.frame_num].copy(), [_box], None))
                if face_embedding_list is None or len(face_embedding_list) == 0:
                    continue
                current_face_vec = face_embedding_list[0].detach().numpy()
            except Exception:
                continue

            try:
                predicted_id = self.system.svc.predict([current_face_vec])[0]
                registered_embeddings = self.system.state.features_dict.get(predicted_id, [])
                if not registered_embeddings:
                    continue
                total_similarity = sum(cosine_similarity(current_face_vec, emb) for emb in registered_embeddings)
                confidence = total_similarity / len(registered_embeddings)
            except Exception:
                continue

            if confidence >= self.CONFIDENCE_THRESHOLD:
                person_id = predicted_id
                state = self.recognition_states.get(person_id, {'score': 0, 'last_seen': now, 'history': []})

                state['score'] += confidence
                state['last_seen'] = now
                state['history'].append((confidence, now))
                self.recognition_states[person_id] = state

                display_lines = []
                last_three_history = state['history'][-3:]
                for conf, ts in last_three_history:
                    time_str = datetime.fromtimestamp(ts, self.TIMEZONE).strftime('%H:%M:%S')
                    display_lines.append(f"{conf:.1%} @ {time_str}")
                self.system.state.display_history[self.frame_num] = display_lines

                if state['score'] >= self.SUCCESS_SCORE_THRESHOLD:
                    log_time = datetime.now(self.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')

                    history_log = " | ".join([
                        f"{conf:.2f} @ {datetime.fromtimestamp(ts, self.TIMEZONE).strftime('%H:%M:%S')}"
                        for conf, ts in state['history']
                    ])
                    print(f"--- [{log_time}][成功 Cam {self.frame_num}] {person_id} 辨識成功! ---")
                    print(f"    - 最終分數: {state['score']:.2f} (>{self.SUCCESS_SCORE_THRESHOLD})")
                    print(f"    - 分數組成: [ {history_log} ]")

                    self.system.state.same_people[self.frame_num] = 1
                    self._update_display_state(person_id)

                    del self.recognition_states[person_id]

            if time.time() - last_warmup_time > 10 and self.frame_num == 0:
                try:
                    _ = self.system.resnet(dummy_input)
                    last_warmup_time = time.time()
                except:
                    pass

    def terminate(self):
        self.stop_threads = True
