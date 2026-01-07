
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
        """
        self.txt = ""
        self.filename = ""
        self.path = os.path.join(os.path.dirname(__file__), "../voice/")
        self.play = False
        self.stop_threads = False
        self.mixer_initialized = False # Mixer 初始化成功旗標

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # 啟動背景執行緒執行 speak()
        th = threading.Thread(target=self.speak, name="speak")
        th.daemon = True
        th.start()

    def speak(self):
        """
        語音合成與播放功能主循環。
        在背景執行緒中執行，會先嘗試初始化音訊設備。
        """
        # --- 在背景執行緒中執行一次性初始化 ---
        try:
            mixer.init()
            self.mixer_initialized = True
            LOGGER.info("音訊設備在背景執行緒中初始化成功。")
        except Exception as e:
            LOGGER.error(f"背景初始化音訊設備失敗: {e}。語音播報功能已被永久停用。")
            # 初始化失敗，直接結束這個執行緒
            self.stop_threads = True
            return

        # --- 主循環 ---
        while not self.stop_threads:
            if self.play:
                if self.mixer_initialized:
                    LOGGER.info(f"偵測到播放請求: '{self.txt}'")
                    filename = self.filename + ".mp3"
                    full_path = os.path.join(self.path, filename)
                    try:
                        # 在載入新音訊前，強制停止所有正在播放的音訊，確保即時性
                        try:
                            mixer.music.stop()
                            mixer.quit()
                        except:
                            pass
                        
                        mixer.init()

                        if not os.path.isfile(full_path):
                            LOGGER.info(f"語音檔不存在，正在使用 gTTS 產生: {full_path}")
                            tts = gTTS(text=self.txt, lang='zh-tw')
                            tts.save(full_path)
                            LOGGER.info("gTTS 語音檔產生成功。")

                        LOGGER.info(f"正在載入並播放語音檔: {full_path}")
                        mixer.music.load(full_path)
                        mixer.music.set_volume(1.0) # 強制設定音量為最大
                        mixer.music.play(1)
                        LOGGER.info("語音播放指令已送出。")
                    except Exception as e:
                        LOGGER.error(f"語音播報時發生錯誤: {e}", exc_info=True)
                
                # 無論成功或失敗，都重置播放旗標
                self.play = False
            
            time.sleep(0.01)

    def is_busy(self):
        """
        檢查目前是否正在播放音訊。
        """
        if self.mixer_initialized:
            try:
                return mixer.music.get_busy()
            except:
                return False
        return False

    def terminate(self):
        """
        終止語音播放的背景執行緒，並防止播放。
        將 stop_threads 設為 True,speak() 將在下一輪中停止迴圈。
        """
        self.stop_threads = True
        self.play = False