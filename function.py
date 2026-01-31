
import datetime
import json
from PVMS_Library import config
import os
import threading
import time
import requests
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance

from init.log import LOGGER

with open(os.path.join(os.path.dirname(__file__), "config.json"), "r", encoding="utf-8") as json_file:
    CONFIG = json.load(json_file)

API = config.API(str(CONFIG["Server"]["API_url"]), int(CONFIG["Server"]["location_ID"]))

def remove_old_files(directory, n=2000, m=100):
    """
    移除指定資料夾中最舊的 m 個檔案（當檔案總數超過 n 時）。

    :param directory: 要清理的資料夾路徑
    :param n: 檔案總數超過此數量才啟動清理
    :param m: 要刪除的最舊檔案數量
    """
    # 取得資料夾底下所有檔案的完整路徑
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # 如果檔案數量超過 n，進行處理
    if len(files) > n:
        # 按照檔案的最後修改時間進行排序
        files.sort(key=lambda x: os.path.getmtime(x))
        
        # 刪除最舊的 m 個檔案
        for file_to_delete in files[:m]:
            try:
                os.remove(file_to_delete)
                print(f"Deleted: {file_to_delete}")
            except Exception as e:
                print(f"Failed to delete {file_to_delete}: {e}")
    else:
        print("No need to delete files.")

def check_in_out(system, staff_name, staff_id, camera_num, n, confidence):
    """
    根據攝影機與時間控制簽到/簽離邏輯，並執行 API 上傳與語音播放。

    :param system: 系統物件(包含狀態 state 與 speaker)
    :param staff_name: 人員名稱
    :param staff_id: 人員 ID
    :param camera_num: 攝影機編號(0 為進入, 1 為離開)
    :param n: 是否為單鏡頭模式(True=單鏡頭自動切換, False=雙鏡頭)
    :param confidence: 辨識信賴度
    :return: 是否為離開狀態(0=非離開, 1=已離開超過2秒)
    """
    leave = 0
    if not staff_id in system.state.check_time.keys():
        system.state.check_time[staff_id] = [True, 0]

    now = time.time()

    # [2026-01-08 修正] 移除全域語音鎖，恢復針對個人的短防抖 (Debounce)
    # 解決 "A正在播音時，B被擋住導致有圖無聲/沒刷入" 的問題
    # 設定 2.5 秒個人防抖：
    # 1. 確保同一人在 "請進入/請離開" 語音期間不被重複觸發
    # 2. 確保不同人之間可以隨時插播與排隊，不再互相干擾
    # if (now - system.state.check_time[staff_id][1]) < 2.5:
    #     return leave
    
    # 簽到/簽離邏輯判斷
    is_check_in_action = False
    is_check_out_action = False
    
    # [2026-01-27 Feature] Time-based Schedule (Overrides standard logic)
    schedule_conf = CONFIG.get("Schedule", {})
    schedule_active = False
    
    if schedule_conf.get("enabled", False):
        try:
            # Use local time
            now_dt = datetime.datetime.now()
            now_time = now_dt.time()
            
            is_in_period = False
            
            # Support multiple periods (Prioritize new list format)
            periods = schedule_conf.get("in_periods", [])
            if not periods:
                # Fallback to legacy single period
                start_str = schedule_conf.get("in_start", "06:00")
                end_str = schedule_conf.get("in_end", "17:00")
                periods = [{"start": start_str, "end": end_str}]
            
            for period in periods:
                start_str = period.get("start", "00:00")
                end_str = period.get("end", "00:00")
                
                start_time = datetime.datetime.strptime(start_str, "%H:%M").time()
                end_time = datetime.datetime.strptime(end_str, "%H:%M").time()
                
                # Check if current time is within this period
                if start_time <= end_time:
                    if start_time <= now_time <= end_time:
                        is_in_period = True
                        break
                else:
                    # Cross-midnight case
                    if start_time <= now_time or now_time <= end_time:
                        is_in_period = True
                        break
                
            if is_in_period:
                is_check_in_action = True
            else:
                is_check_out_action = True
                
            schedule_active = True
        except Exception as e:
            LOGGER.error(f"Schedule Logic Error: {e}")

    if not schedule_active:
        if n:  # 單鏡頭模式：根據狀態自動判斷
            if system.state.check_time[staff_id][0]:
                is_check_in_action = True
            else:
                is_check_out_action = True
        else:  # 雙鏡頭模式：根據攝影機編號判斷
            if camera_num == 0:
                is_check_in_action = True
            elif camera_num == 1:
                is_check_out_action = True

    # 執行簽到
    if is_check_in_action:
        log_metrics(staff_name, 0, confidence) # Log as check-in with confidence
        async_api_call(
            func=API.face_recognition_in,
            args=(staff_id,),
            callback=log_api_result,
            system=system,
            staff_id=staff_id,
            action='in'
        )
        # 語音播報：簽到成功為最高優先權 (Priority=1)，可插播提示語音，但會排隊等待其他簽到語音
        system.speaker.say(f"{staff_name}{CONFIG['say']['in']}", staff_name + "_in", priority=1, token=staff_id)

    # 執行簽離
    elif is_check_out_action:
        log_metrics(staff_name, 1, confidence) # Log as check-out with confidence
        async_api_call(
            func=API.face_recognition_out,
            args=(staff_id,),
            callback=log_api_result,
            system=system,
            staff_id=staff_id,
            action='out'
        )
        # 語音播報：簽離成功為最高優先權 (Priority=1)
        system.speaker.say(f"{staff_name}{CONFIG['say']['out']}", staff_name + "_out", priority=1, token=staff_id)

    if CONFIG.get("excel_api_enabled", False) and "demosite" in CONFIG["Server"]["API_url"]:
        threading.Timer(1, check_in_out_excel, (staff_name,)).start()

    # 開門控制
    if CONFIG["door"] != "0":
        threading.Timer(0, open_door).start()

    # 原本的離開判斷保留，雖在主流程未被使用，但保持結構完整
    # [2026-01-20 Fix] 防止 Race Condition 導致 KeyError
    # 若在執行到此處時 staff_id 被其他執行緒(如 clear_leave_employee)刪除，則視為初始狀態
    last_time = system.state.check_time.get(staff_id, [True, 0])[1]
    if (now - last_time) >= 2:
        leave = 1

    return leave

def check_in_out_excel(staff_name):
    """
    發送 HTTP POST 請求至 Excel attendance API 用於 demo 匯入。

    :param staff_name: 簽到人員的名字
    """
    url = f"http://{CONFIG['ip_set']['ip_address']}:8080/attendance-record"
    data = {"name": staff_name, "time": datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"), "location": "gini"}
    
    try:
        response = requests.post(url, json=data, timeout=5) # Timeout set to 5 seconds
        print("excel", response.status_code)
    except requests.exceptions.RequestException as e:
        print(f"Excel匯入失敗: {e}")
        LOGGER.error(f"Excel匯入失敗: {e}")

def open_door():
    """
    透過設定的門禁 URL 觸發開門操作，並寫入 log。
    """
    r = requests.get(CONFIG["door"])
    LOGGER.info(f"{time.time()} : 開門 {r}")
        
def clear_leave_employee(system, staff_id):
    """
    在延遲一段時間後清除人員的 check_in/out 記錄，用於重置簽到狀態。

    :param system: 系統物件，需包含 state.check_time 結構
    :param staff_id: 人員 ID
    """
    if staff_id in system.state.check_time.keys():
        del system.state.check_time[staff_id]
        print(f"clear {staff_id}")
        LOGGER.info(f"clear {staff_id}")

def log_metrics(employee, camera_num, confidence=None):
    """
    將簽到或簽離的事件記錄進 LOGGER 並列印。

    :param employee: 人員名稱
    :param camera_num: 攝影機編號(0=簽到, 1=簽離)
    :param confidence: 辨識信賴度 (可選)
    """
    inoutType = ""
    log_cam_num = 0
    if camera_num == 0:
        inoutType = "進入"
        log_cam_num = 0
    elif camera_num == 1:
        inoutType = "離開"
        log_cam_num = 1
    
    conf_str = f", 信賴度: {confidence:.2%}" if confidence is not None else ""
    log_message = f"攝影機編號:{log_cam_num}, 人員:{employee} {inoutType}{conf_str}"
    print(log_message)
    LOGGER.info(log_message)


import threading

def diagnose_network():
    """
    診斷網路連線狀況，用於上傳失敗時釐清原因。
    測試目標: 
    1. 外部網路 (Google DNS 8.8.8.8)
    2. 專案伺服器 API
    """
    results = []
    
    # 1. Ping Google DNS (8.8.8.8) - 測試外網連通性
    # -c 1: 發送一次
    # -W 2: 等待2秒
    response = os.system("ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1")
    if response == 0:
        results.append("外部網路(8.8.8.8): 連通")
    else:
        results.append("外部網路(8.8.8.8): 無法連線")
        
    # 2. 測試伺服器連線 (使用 requests)
    server_url = CONFIG.get("Server", {}).get("API_url", "")
    if server_url:
        try:
            # 只測試連線，不在此乎回傳內容，設定短 timeout
            requests.get(server_url, timeout=3)
            results.append(f"伺服器({server_url}): 連通")
        except Exception as e:
            results.append(f"伺服器({server_url}): 無法連線 ({e})")
    else:
        results.append("伺服器URL未設定")
        
    return ", ".join(results)

def async_api_call(func, args=(), callback=None, max_retries=20, retry_delay=0.5, system=None, staff_id=None, action=None):
    """
    非同步執行 API 呼叫，並在成功後更新系統狀態。
    
    :param func: 要執行的函數
    :param args: 傳給函數的參數 (tuple)
    :param callback: 成功時執行 callback(result)
    :param max_retries: 最大重試次數
    :param retry_delay: 每次重試間隔秒數
    :param system: 全域系統物件
    :param staff_id: 要更新狀態的人員ID
    :param action: 'in' 或 'out'，決定如何更新狀態
    """
    def task():
        for attempt in range(1, max_retries + 1):
            try:
                result = func(*args)

                if result in [201, 202]:
                    LOGGER.info(f"[{func.__name__}] 成功，第 {attempt} 次取得 {result}")
                    if callback:
                        callback(result)
                    
                    # 在API成功後才更新狀態
                    if system and staff_id and action:
                        now = time.time()
                        try:
                            if action == 'in':
                                system.state.check_time[staff_id] = [False, now]
                            elif action == 'out':
                                system.state.check_time[staff_id][1] = now
                                threading.Timer(5, clear_leave_employee, (system, staff_id)).start()
                            LOGGER.info(f"人員 {staff_id} 狀態已在API成功後更新為 {action}")
                        except Exception as e:
                            LOGGER.error(f"API成功後更新人員 {staff_id} 狀態失敗: {e}")
                    return
                else:
                    LOGGER.warning(f"[{func.__name__}] 嘗試 {attempt} 次：API 回傳代碼 {result} (非預期的 201 或 202)")
                    time.sleep(retry_delay)

            except Exception as e:
                LOGGER.warning(f"[{func.__name__}] 呼叫失敗，第 {attempt} 次：{e}")
                time.sleep(retry_delay)

        LOGGER.error(f"[{func.__name__}] 最多重試 {max_retries} 次仍未收到成功回應。")
        
        # 上傳失敗後執行網路診斷
        network_status = diagnose_network()
        LOGGER.error(f"[{func.__name__}] 上傳失敗網路診斷: {network_status}")

    threading.Thread(target=task).start()

def log_api_result(res):
    """
    將 API 回傳結果寫入日誌與列印。

    :param res: API 回傳的結果(通常是 HTTP status code)
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now_str} 上傳資料庫結果: {res}")
    LOGGER.info(f"{now_str} 上傳資料庫結果:" + str(res))

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def crop_face_without_forehead(image, box, points, image_size=160):
    """
    Option SMALL (Method J) Implementation:
    - Square Crop (2.0 * Eye Distance)
    - Centered on Eyes/Nose
    - Square Padding (Black Background)
    - No Mask
    - Area Resize
    """
    # Convert PIL to Numpy (RGB)
    img_np = np.array(image)
    
    # Points: 0=LE, 1=RE, 2=Nose
    le = points[0]
    re = points[1]
    nose = points[2]
    
    # Calculate Geometry
    eye_dist = np.linalg.norm(le - re)
    
    eye_center = (le + re) / 2
    t_center_x = (eye_center[0] + nose[0]) / 2
    t_center_y = (eye_center[1] + nose[1]) / 2
    
    # Shift center down 0.2 * eye_dist
    t_center_y += eye_dist * 0.2
    
    # Crop Size: 2.0 * Eye Distance
    size = int(eye_dist * 2.0)
    
    x1 = int(t_center_x - size/2)
    y1 = int(t_center_y - size/2)
    x2 = x1 + size
    y2 = y1 + size
    
    # Square Padding Logic
    h, w = img_np.shape[:2]
    square_img = np.zeros((size, size, 3), dtype=np.uint8)
    
    src_x1 = max(0, x1); src_y1 = max(0, y1)
    src_x2 = min(w, x2); src_y2 = min(h, y2)
    
    dst_x1 = src_x1 - x1; dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        square_img[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]
        
    # Resize (Area Interpolation)
    img_resized = cv2.resize(square_img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    # To Tensor & Standardize
    face_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    standardized_face = fixed_image_standardization(face_tensor)
    
    return standardized_face

# [2026-01-19 Feature] Part-Based Cropping
def crop_and_pad(img, cx, cy, w, h, target_size=160):
    """Helper to crop a region and resize it to target size."""
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    
    # Crop (PIL handles out of bounds by padding with 0 if using proper method, 
    # but crop() just clamps. We want context.)
    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return crop

def get_parts_crop(image_pil, landmarks):
    """
    Crop Eye, Nose, Mouth regions based on MediaPipe landmarks.
    Returns:
        parts_tensors (dict): {'eye': tensor, ...} ready for ResNet.
        parts_coords (dict): {'eye': [cx, cy, w, h], ...} for logging/debugging.
    
    Landmarks indices (5 points from MediaPipeHandler):
    0=L_Eye, 1=R_Eye, 2=Nose, 3=L_Mouth, 4=R_Mouth
    """
    parts_tensors = {}
    parts_coords = {}
    
    # [2026-01-20 New Feature] T-Zone Long Crop (Eyebrows + Eyes + Nose + Philtrum)
    # Replaces individual Eye/Nose/Mouth checks for better stability against expression/glasses.
    # Center: Midpoint between Eye-Center and Nose
    eye_center_x = (landmarks[0][0] + landmarks[1][0]) / 2
    eye_center_y = (landmarks[0][1] + landmarks[1][1]) / 2
    nose_x, nose_y = landmarks[2]
    
    eye_dist = np.linalg.norm(landmarks[0] - landmarks[1])
    
    t_center_x = (eye_center_x + nose_x) / 2
    t_center_y = (eye_center_y + nose_y) / 2
    
    # Shift center down slightly for "Long" version to include Philtrum without cutting forehead
    t_long_cy = t_center_y + eye_dist * 0.2
    
    # Dimensions: 2.0x EyeDist Width, 3.0x EyeDist Height
    crop_w = eye_dist * 2.0
    crop_h = eye_dist * 3.0
    
    parts_tensors['t_zone'] = _process_part_tensor(crop_and_pad(image_pil, t_center_x, t_long_cy, crop_w, crop_h))
    parts_coords['t_zone'] = [float(x) for x in [t_center_x, t_long_cy, crop_w, crop_h]]
    
    return parts_tensors, parts_coords

def _process_part_tensor(img_pil):
    """Standardize a part crop to tensor."""
    # No extra sharpening for parts (keep it raw)
    face_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()
    processed_tensor = (face_tensor - 127.5) / 128.0
    return processed_tensor