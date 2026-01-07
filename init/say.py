
#寫成函式
import time
from gtts import gTTS
from pygame import mixer
import os, threading
import queue
from init.log import LOGGER

class Say_:
    """
    語音播報類別，用於將文字轉換為語音(使用 Google TTS)並播放。
    改用 Queue 指令佇列與 Generation 版本號機制，解決指令覆蓋(Race Condition)問題，
    並增強對 ALSA 音訊設備錯誤的容錯能力。
    """
    def __init__(self):
        """
        建構子，初始化語音播放控制項與背景執行緒。
        """
        self.queue = queue.Queue()
        self.generation = 0 # 版本號，用於標記指令是否有效
        self.gen_lock = threading.Lock() # 保護版本號的鎖

        self.path = os.path.join(os.path.dirname(__file__), "../voice/")
        self.stop_threads = False
        self.mixer_initialized = False 

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # 啟動背景執行緒執行 speak()
        th = threading.Thread(target=self.speak, name="speak")
        th.daemon = True
        th.start()

    def say(self, text, filename):
        """
        發送語音播報請求 (Thread-Safe)。
        此方法會將指令放入佇列，保證不丟失。
        """
        with self.gen_lock:
            # 取得當前有效的版本號
            current_gen = self.generation
        
        # 將請求放入佇列 (版本號, 文字, 檔名)
        self.queue.put((current_gen, text, filename))

    def stop(self):
        """
        強制停止目前的音訊播放，並作廢佇列中所有未執行的指令 (插播用)。
        """
        with self.gen_lock:
            self.generation += 1 # 提升版本號，使舊指令失效
        
        # 清空佇列 (雖然提升版本號已足夠讓舊指令失效，但清空可減少無效迴圈)
        with self.queue.mutex:
            self.queue.queue.clear()

        if self.mixer_initialized:
            try:
                mixer.music.stop()
            except Exception as e:
                LOGGER.error(f"強制停止播放失敗: {e}")

    def speak(self):
        """
        語音合成與播放功能主循環。
        """
        # --- 在背景執行緒中執行一次性初始化 ---
        try:
            # 嘗試初始化，若失敗不立即退出，留待迴圈內重試
            mixer.init()
            self.mixer_initialized = True
            LOGGER.info("音訊設備在背景執行緒中初始化成功。")
        except Exception as e:
            LOGGER.error(f"初次初始化音訊設備失敗 (將嘗試重試): {e}")

        # --- 主循環 ---
        while not self.stop_threads:
            try:
                # 1. 從佇列取出指令 (Blocking with timeout)
                # 設定 timeout 是為了讓執行緒有機會檢查 stop_threads
                gen, text, filename_base = self.queue.get(timeout=0.1)
                
                # 2. 檢查版本號 (過期指令直接丟棄)
                with self.gen_lock:
                    if gen != self.generation:
                        LOGGER.info(f"忽略過期語音指令: {text} (Gen {gen} != {self.generation})")
                        continue

                LOGGER.info(f"處理語音請求: '{text}'")
                filename = filename_base + ".mp3"
                full_path = os.path.join(self.path, filename)
                
                try:
                    # 3. 音訊子系統重置 (針對 ALSA 錯誤的對策)
                    # 頻繁的 init/quit 雖然重，但能有效解決 "Unknown PCM default"
                    try:
                        mixer.music.stop()
                        mixer.quit()
                    except:
                        pass
                    
                    # 嘗試重新初始化 mixer
                    mixer.init()
                    self.mixer_initialized = True

                    # 4. 準備語音檔
                    if not os.path.isfile(full_path):
                        LOGGER.info(f"語音檔不存在，正在使用 gTTS 產生: {full_path}")
                        tts = gTTS(text=text, lang='zh-tw')
                        tts.save(full_path)
                        LOGGER.info("gTTS 語音檔產生成功。")

                    # 5. 二次檢查版本號 (避免生成檔案期間被插播)
                    with self.gen_lock:
                        if gen != self.generation:
                            LOGGER.info(f"播放前被中斷，取消播放: {text}")
                            continue

                    # 6. 播放
                    LOGGER.info(f"正在載入並播放語音檔: {full_path}")
                    mixer.music.load(full_path)
                    mixer.music.set_volume(1.0)
                    mixer.music.play(1)
                    
                    # 等待播放完成或被中斷 (避免 mixer.quit 殺太快)
                    # 我們不使用 while busy loop 等待，因為這會卡住插播
                    # 這裡只負責 "觸發播放"，實際聲音由 SDL thread 處理
                    # 但因為我們下一輪迴圈會 mixer.quit()，所以必須確保聲音播出去
                    # 妥協方案：讓 mixer 活著直到 busy 為止，或被 stop
                    start_wait = time.time()
                    while mixer.music.get_busy() and (time.time() - start_wait < 10): # 最多等10秒
                        # 檢查是否被插播 (stop 呼叫)
                        with self.gen_lock:
                            if gen != self.generation:
                                mixer.music.stop()
                                break
                        time.sleep(0.1)
                    
                except Exception as e:
                    LOGGER.error(f"語音播報執行失敗 (ALSA/Pygame Error): {e}", exc_info=True)
                    self.mixer_initialized = False # 標記失敗，下次迴圈重試

            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.error(f"語音佇列處理發生未預期錯誤: {e}")
                time.sleep(1)

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