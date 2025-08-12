import datetime
import json
from PVMS_Library import config
import os
import threading
import time
import requests

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

def check_in_out(system, staff_name, staff_id, camera_num, n):
    """
    根據攝影機與時間控制簽到/簽離邏輯，並執行 API 上傳與語音播放。

    :param system: 系統物件(包含狀態 state 與 speaker)
    :param staff_name: 人員名稱
    :param staff_id: 人員 ID
    :param camera_num: 攝影機編號(0 為進入, 1 為離開)
    :param n: 強制簽到簽離模式(True=不管狀態都觸發)
    :return: 是否為離開狀態(0=非離開, 1=已離開超過2秒)
    """
    leave = 0
    if not staff_id in system.state.check_time.keys():
        system.state.check_time[staff_id] = [True, 0]

    now = time.time()

    if (now - system.state.check_time[staff_id][1]) >= 10:
        if camera_num == 0 and (not n or system.state.check_time[staff_id][0]):
            print(f"攝影機編號:{camera_num}, 人員:{staff_name} 進入")
            
            # ✅ 非同步上傳簽到資料
            async_api_call(
                func=API.face_recognition_in,
                args=(staff_id,),
                callback=log_api_result
            )

            log_metrics(staff_name, camera_num)
            system.speaker.txt = f"{staff_name}{CONFIG['say']['in']}"
            system.speaker.filename = staff_name + "_in"
            system.speaker.play = True

            try:
                system.state.check_time[staff_id] = [False, now]
            except:
                print("簽入錯誤")
                LOGGER.info("簽入錯誤")

        elif camera_num == 1 and (not n or not system.state.check_time[staff_id][0]):
            print(f"攝影機編號:{camera_num}, 人員:{staff_name} 離開")

            # ✅ 非同步上傳簽出資料
            async_api_call(
                func=API.face_recognition_out,
                args=(staff_id,),
                callback=log_api_result
            )

            log_metrics(staff_name, camera_num)
            system.speaker.txt = f"{staff_name}{CONFIG['say']['out']}"
            system.speaker.filename = staff_name + "_out"
            system.speaker.play = True

            try:
                system.state.check_time[staff_id][1] = now
                threading.Timer(5, clear_leave_employee, (system, staff_id)).start()
            except:
                print(f"{staff_id}尚未簽入")
                LOGGER.info(f"{staff_id}尚未簽入")

        if "demosite" in CONFIG["Server"]["API_url"] and system.speaker.play:
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
    data = {}
    data["name"] = staff_name
    data["time"] = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    data["location"] = "gini"
    print("excel", requests.post(url,json=data,timeout=1000))

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

def log_metrics(employee, camera_num):
    """
    將簽到或簽離的事件記錄進 LOGGER。

    :param employee: 人員名稱
    :param camera_num: 攝影機編號(0=簽到, 1=簽離)
    """
    inoutType = ""
    if camera_num == 0:
        inoutType = "簽到"
    elif camera_num == 1:
        inoutType = "簽離"
    LOGGER.info(f"{time.time()} : {employee}  {inoutType}")

import threading

def async_api_call(func, args=(), callback=None, max_retries=20, retry_delay=0.5):
    """
    非同步執行 API 呼叫，直到回傳 201 或超過重試次數。
    
    :param func: 要執行的函數
    :param args: 傳給函數的參數 (tuple)
    :param callback: 成功時執行 callback(result)
    :param max_retries: 最大重試次數
    :param retry_delay: 每次重試間隔秒數
    """
    def task():
        for attempt in range(1, max_retries + 1):
            try:
                result = func(*args)

                if result == 201:
                    LOGGER.info(f"[{func.__name__}] 成功，第 {attempt} 次取得 201")
                    if callback:
                        callback(result)
                    return
                else:
                    LOGGER.warning(f"[{func.__name__}] 嘗試 {attempt} 次：回傳非 201（實際為 {result}）")
                    time.sleep(retry_delay)

            except Exception as e:
                LOGGER.warning(f"[{func.__name__}] 呼叫失敗，第 {attempt} 次：{e}")
                time.sleep(retry_delay)

        LOGGER.error(f"[{func.__name__}] 最多重試 {max_retries} 次仍未收到 201。")

    threading.Thread(target=task).start()

def log_api_result(res):
    """
    將 API 回傳結果寫入日誌與列印。

    :param res: API 回傳的結果(通常是 HTTP status code)
    """
    print("上傳資料庫結果:", res)
    LOGGER.info("上傳資料庫結果:" + str(res))
