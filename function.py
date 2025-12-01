import datetime
import json
from PVMS_Library import config
import os
import threading
import time
import requests
import torch
import numpy as np
from PIL import Image

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

    if (now - system.state.check_time[staff_id][1]) >= 10:
        # 簽到/簽離邏輯判斷
        is_check_in_action = False
        is_check_out_action = False

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
            # 語音播報冷卻
            if now - system.state.last_speak_time.get(staff_id, 0) >= 5:
                system.speaker.txt = f"{staff_name}{CONFIG['say']['in']}"
                system.speaker.filename = staff_name + "_in"
                system.speaker.play = True
                system.state.last_speak_time[staff_id] = now

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
            # 語音播報冷卻
            if now - system.state.last_speak_time.get(staff_id, 0) >= 5:
                system.speaker.txt = f"{staff_name}{CONFIG['say']['out']}"
                system.speaker.filename = staff_name + "_out"
                system.speaker.play = True
                system.state.last_speak_time[staff_id] = now

        if CONFIG.get("excel_api_enabled", False) and "demosite" in CONFIG["Server"]["API_url"]:
            threading.Timer(1, check_in_out_excel, (staff_name,)).start()

        # 開門控制
        if CONFIG["door"] != "0":
            threading.Timer(0, open_door).start()

    elif (now - system.state.check_time[staff_id][1]) >= 2:
        leave = 1

    return leave

def check_in_out_excel(staff_name):
    """
    發送 HTTP POST 請求至 Excel attendance API 用於 demo 匯入。

    :param staff_name: 簽到人員的名字
    """
    url = "http://192.168.31.130:8080/attendance-record"
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
                    LOGGER.warning(f"[{func.__name__}] 嘗試 {attempt} 次：回傳非 {result}（應為 201 或 202）")
                    time.sleep(retry_delay)

            except Exception as e:
                LOGGER.warning(f"[{func.__name__}] 呼叫失敗，第 {attempt} 次：{e}")
                time.sleep(retry_delay)

        LOGGER.error(f"[{func.__name__}] 最多重試 {max_retries} 次仍未收到成功回應。")

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
    Crops the face from an image, excluding the forehead, based on eye landmarks.

    Args:
        image (PIL.Image): The input image.
        box (list or np.ndarray): The bounding box of the face [x1, y1, x2, y2].
        points (np.ndarray): Facial landmarks, where points[0] is the left eye and points[1] is the right eye.
        image_size (int): The desired output image size.

    Returns:
        torch.Tensor: A tensor of the cropped, resized, and standardized face image.
    """
    # Calculate the y-coordinate of the eyes' center
    eye_y_center = (points[0][1] + points[1][1]) / 2
    
    # Get the original bounding box coordinates
    x1, y1, x2, y2 = box
    
    # Calculate the height of the face box and the distance from the top to the eyes
    box_height = y2 - y1
    eye_to_top_dist = eye_y_center - y1

    # Define the new top of the crop area to be a bit above the eyes
    # We use a fraction of the eye_to_top_dist to set the new top margin
    new_y1 = eye_y_center - (eye_to_top_dist * 0.4) # Crop closer to the eyes
    
    # Ensure the new y1 is within the original image bounds
    new_y1 = max(0, new_y1)

    # Create the new bounding box
    new_box = [x1, new_y1, x2, y2]
    
    # Crop the image using the new bounding box
    img_cropped = image.crop(new_box)
    
    # Resize to the target square size
    img_resized = img_cropped.resize((image_size, image_size), Image.BILINEAR)
    
    # Convert to tensor and standardize
    face_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
    standardized_face = fixed_image_standardization(face_tensor)
    
    return standardized_face