
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
        self.generation = 0 
        self.gen_lock = threading.Lock()
        
        # 用於追蹤當前正在播放的語音優先級
        self.current_priority = 0 # 0=Idle, 1=High, 2=Normal
        self.priority_lock = threading.Lock()

        self.path = os.path.join(os.path.dirname(__file__), "../voice/")
        self.stop_threads = False
        self.mixer_initialized = False 

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # 啟動背景執行緒執行 speak()
        th = threading.Thread(target=self.speak, name="speak")
        th.daemon = True
        th.start()

    def say(self, text, filename, priority=2):
        """
        發送語音播報請求 (Thread-Safe)。
        priority: 1=High (簽到/簽離，必播，可插播Normal), 2=Normal (提示，忙碌即丟棄)
        """
        with self.priority_lock:
            is_busy = self.current_priority != 0
            current_p = self.current_priority

        # 邏輯 1: Normal 語音 (提示類)
        if priority == 2:
            # 如果系統正忙 (不論是 High 或 Normal)，直接丟棄，不排隊
            if is_busy:
                LOGGER.debug(f"語音忙碌，丟棄提示語音: {text}")
                return
            # 如果閒置，放入佇列
            self._enqueue(text, filename, priority)

        # 邏輯 2: High 語音 (簽到類)
        elif priority == 1:
            if current_p == 2:
                # 正在播 Normal -> 強制打斷 (Preempt)，然後自己播
                LOGGER.info(f"強制中斷提示語音，插播重要訊息: {text}")
                self._stop_current()
                self._enqueue(text, filename, priority)
            else:
                # 正在播 High 或 閒置 -> 進 Queue 排隊 (High 互不打斷，依序播放)
                # 限制 Queue 長度，避免堆積過多
                if self.queue.qsize() >= 2:
                    try:
                        self.queue.get_nowait() # 丟棄最舊的
                        LOGGER.warning("語音佇列過滿，丟棄舊指令")
                    except:
                        pass
                self._enqueue(text, filename, priority)

    def _enqueue(self, text, filename, priority):
        with self.gen_lock:
            gen = self.generation
        self.queue.put((gen, text, filename, priority))

    def _stop_current(self):
        """內部方法：強制停止目前播放 (提升版本號)"""
        with self.gen_lock:
            self.generation += 1
        
        if self.mixer_initialized:
            try:
                mixer.music.stop()
            except:
                pass

    # 移除公開的 stop() 方法，因為邏輯已整合進 say()
    # def stop(self): ...

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
                # 1. 從佇列取出指令
                gen, text, filename_base, priority = self.queue.get(timeout=0.1)
                
                # 2. 檢查版本號 (過期指令直接丟棄)
                with self.gen_lock:
                    if gen != self.generation:
                        LOGGER.debug(f"忽略過期語音: {text}")
                        continue

                # 設定當前優先級狀態
                with self.priority_lock:
                    self.current_priority = priority

                LOGGER.info(f"播放語音 (P{priority}): '{text}'")
                filename = filename_base + ".mp3"
                full_path = os.path.join(self.path, filename)
                
                try:
                    # 3. 音訊子系統檢查與重置 (Lazy Re-init)
                    # 只有在未初始化或先前發生錯誤時才執行 init
                    if not self.mixer_initialized:
                        try:
                            mixer.quit()
                            mixer.init()
                            self.mixer_initialized = True
                            LOGGER.info("音訊子系統已(重新)初始化。")
                        except Exception as e:
                            LOGGER.error(f"音訊初始化失敗: {e}")
                            # 稍後重試，跳過本次播放
                            time.sleep(1)
                            continue

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
                    
                    # 等待播放完成或被中斷
                    start_wait = time.time()
                    while mixer.music.get_busy() and (time.time() - start_wait < 10): 
                        # 檢查是否被插播 (stop 呼叫)
                        with self.gen_lock:
                            if gen != self.generation:
                                mixer.music.stop()
                                break
                        time.sleep(0.1)
                    
                except Exception as e:
                    LOGGER.error(f"語音播報執行失敗 (ALSA/Pygame Error): {e}", exc_info=True)
                    self.mixer_initialized = False 
                
                finally:
                    # 播放結束 (或失敗)，重置優先級狀態為 Idle
                    with self.priority_lock:
                        self.current_priority = 0

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