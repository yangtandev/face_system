import queue
import subprocess
import threading
import time
import cv2
import numpy as np  # 用於創建黑畫面
from urllib.parse import urlparse

class VideoCapture:
    """
    Custom VideoCapture class to handle RTSP connection failures and auto-reconnect.
    自定義的 RTSP 影像擷取類別，支援自動重連與讀取錯誤時填黑畫面。
    """

    def __init__(self, rtsp_url, retries=5, delay=5):
        """
        Constructor 初始化 RTSP 連線物件，並啟動讀取執行緒。

        Parameters:
        rtsp_url (str): RTSP 串流位址。
        retries (int): 最多重試次數（目前未使用，可擴充）。
        delay (int): 每次重試間隔秒數。
        """
        self.rtsp_url = rtsp_url
        self.retries = retries
        self.delay = delay
        self.cap = None
        self.ret = False
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False
        self.lock = threading.Lock()
        self.count = 0
        # 建立黑畫面 (假設分辨率為 640x480，您可以根據需求調整)
        self.black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        #self._connect()

        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _connect(self):
        """
        嘗試連線到 RTSP 串流。先進行 ping 測試確認 IP 可達，再使用 cv2.VideoCapture 開啟串流。
        若無法連接則每隔 delay 秒重試，直到 stop_threads 被設為 True。
        """
        hostname = urlparse(self.rtsp_url).hostname
        is_localhost = hostname in ("localhost", "127.0.0.1")

        while not self.stop_threads:
            # 無論如何都直接嘗試連線，而不是先 ping
            if True:
                if is_localhost:
                    print(f"Attempting to connect to local RTSP stream: {self.rtsp_url}")
                else:
                    print(f"Attempting to connect to RTSP stream at {hostname}...")

                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    print("Connected to RTSP stream.")
                    return
                else:
                    print("Failed to open RTSP stream with OpenCV. Retrying...")
            
            time.sleep(self.delay)

        # 只有在迴圈是因為 stop_threads=False 結束時才引發例外
        if not self.stop_threads:
            raise Exception("Failed to connect to RTSP stream after multiple attempts.")

    def _reader(self):
        """
        持續從 RTSP 串流讀取影像並放入 queue。若讀取失敗會重連並將 queue 填入黑畫面。
        該函式在初始化時由背景執行緒啟動。
        """
        self._connect()
        while not self.stop_threads:
            try:
                # 提前檢查停止旗標
                if self.stop_threads:
                    break

                with self.lock:
                    # 再次檢查 self.cap 是否有效，避免在無效物件上操作
                    if self.cap is None or not self.cap.isOpened():
                        self.ret = False
                        time.sleep(0.1) # 短暫等待，避免空轉
                        continue

                    ret, frame = self.cap.read()
                    self.ret = ret

                if not ret:
                    # 如果被告知停止，就不要再嘗試重連了
                    if self.stop_threads:
                        break

                    print("Frame read failed. Inserting black frame and reconnecting...")

                    if not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass

                    if self.cap:
                        self.cap.release()
                    self._connect()
                    continue

                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)

            except cv2.error:
                # 在關閉(terminate)過程中，另一執行緒可能會釋放 self.cap，導致 read() 拋出此錯誤。
                # 這是預期行為，所以我們捕捉它並準備結束執行緒。
                if self.stop_threads:
                    print("Reader thread is shutting down gracefully.")
                    break
                else:
                    # 如果不是在關閉過程中發生，則可能是個真正的問題
                    print("An unexpected OpenCV error occurred. Attempting to reconnect...")
                    self.ret = False
                    if self.cap:
                        self.cap.release()
                    time.sleep(self.delay)
                    self._connect()



    def read(self):
        """
        取得最新一張影像(frame)。

        Returns:
        frame (np.ndarray): 最新一張影像，若讀取失敗則回傳黑畫面。
        """
        if not self.ret:
            return self.black_frame
        return self.q.get()

    def ping(self, ip_address):
        """
        嘗試對目標 IP 進行 ping 測試以檢查是否可達。

        Parameters:
        ip_address (str): IP 位址或 RTSP 位址(可包含帳密與 port)

        Returns:
        bool: True 表示 ping 成功,False 表示失敗。
        """
        if "@" in ip_address:
            ip = ip_address.split("@")[-1]
        else:
            ip = ip_address  # 直接傳入 IP 地址的情況
        if ":" in ip:
            ip = ip.split(":")[0]
        try:
            # 使用 sudo 執行 ping 命令
            output = subprocess.check_output(
                ["ping", "-w", "2", ip],
                stderr=subprocess.STDOUT,  # 捕捉標準錯誤輸出
                universal_newlines=True  # 將輸出轉換為字符串
            )

            # 檢查輸出中是否有關鍵字，表示 ping 成功
            if ("Reply from" in output or "回覆自" in output) and \
               ("Received = 4" in output or "已收到 = 4" in output) or \
                "2 received" in output:
                return True  # 表示 ping 成功
            else:
                return False  # 表示 ping 失敗
        except subprocess.CalledProcessError as e:
            #print("Ping failed. Output was:", e.output)
            return False  # 表示命令執行失敗

    def terminate(self):
        """
        優雅地停止讀取執行緒並釋放資源。
        """
        print(f"Terminating camera connection to {self.rtsp_url}...")
        # 1. 設置停止旗標，通知背景執行緒該結束了
        self.stop_threads = True

        # 2. 等待背景執行緒執行完畢 (最多等待 2 秒)
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # 3. 在確認背景執行緒已停止後，才安全地釋放資源
        if self.cap:
            self.cap.release()

        print(f"Camera connection to {self.rtsp_url} terminated.")

if __name__ == "__main__":
    camera = VideoCapture("rtsp://admin:!QAZ87518499@192.168.31.132:554")
    while True:
        frame = camera.read()
        frame = cv2.resize(frame, (1280,720))
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.terminate()
    cv2.destroyAllWindows()
