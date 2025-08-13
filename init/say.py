
#寫成函式
import time
from gtts import gTTS
from pygame import mixer
import os, threading
from init.log import LOGGER

class Say_:
    """
    語音播報類別，用於將文字轉換為語音(使用 Google TTS)並播放,支援非同步控制。
    """
    def __init__(self):
        """
        建構子，初始化語音播放控制項與背景執行緒。
        屬性：
        - self.txt: 要轉成語音的文字
        - self.filename: 儲存 mp3 的檔名（不含副檔名）
        - self.path: 存放 mp3 的資料夾路徑
        - self.play: 是否開始播放語音
        - self.stop_threads: 結束背景執行緒用的旗標
        """
        self.txt = ""
        self.filename = ""
        self.path = os.path.dirname(__file__)+"/../voice/"
        self.play = False
        self.stop_threads = False

        # 啟動背景執行緒執行 speak()
        th = threading.Thread(target=self.speak)
        th.daemon = True             # 设置工作线程为后台运行
        th.start()

    def speak(self):
        """
        語音合成與播放功能主循環：
        - 檢查是否需要播放(self.play 為 True)
        - 若指定的 mp3 檔案不存在，則使用 gTTS 建立新語音檔案
        - 使用 pygame.mixer 播放語音
        - 播放後 self.play 設為 False 以避免重複
        """
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        while not self.stop_threads:
            if self.play:
                mixer.init()
                filename = self.filename + ".mp3" 
                try:
                    if not os.path.isfile(self.path+filename):
                        tts = gTTS(text=self.txt, lang='zh-tw')
                        tts.save(self.path+filename)
                    mixer.music.load(self.path+filename)
                    mixer.music.play(1)
                except Exception as e:
                    LOGGER.warning(f"say : {e}")
                self.play = False
            time.sleep(0.00001)

    def terminate(self):
        """
        終止語音播放的背景執行緒，並防止播放。
        將 stop_threads 設為 True,speak() 將在下一輪中停止迴圈。
        """
        self.stop_threads = True
        self.play = False