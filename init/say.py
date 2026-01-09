
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
        self.force_reinit = False # 新增：強制重置旗標 

        # [2026-01-09] 新增：針對 Token (ID) 的播放狀態追蹤
        self.last_start_time = {} # {token: timestamp}
        self.last_end_time = {}   # {token: timestamp}
        self.status_lock = threading.Lock() # 保護上述字典

        self.last_queued_item = None # (text, timestamp) 用於防止短時間重複入隊
        self.last_preempt_time = 0   # 上次執行插播的時間，用於防止頻繁切斷

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # 啟動背景執行緒執行 speak()
        th = threading.Thread(target=self.speak, name="speak")
        th.daemon = True
        th.start()

    def say(self, text, filename, priority=2, token=None):
        """
        發送語音播報請求 (Thread-Safe)。
        priority: 1=High (簽到/簽離，必播，可插播Normal), 2=Normal (提示，忙碌即丟棄)
        token: 用於追蹤特定對象(ID)的播放狀態，若為None則不追蹤。
        """
        # [2026-01-09] Queue 去重機制：防止短時間內重複指令堆積 (例如連續的"請進入")
        now = time.time()
        if self.last_queued_item:
            last_text, last_time = self.last_queued_item
            # 如果文字相同且間隔小於 1.5 秒，視為重複指令，直接丟棄
            if text == last_text and (now - last_time) < 1.5:
                LOGGER.debug(f"忽略重複語音指令: {text}")
                return
        
        # 更新最後指令記錄
        self.last_queued_item = (text, now)

        with self.priority_lock:
            is_busy = self.current_priority != 0
            current_p = self.current_priority

        # 記錄開始時間 (視為"正在處理")
        if token:
            with self.status_lock:
                self.last_start_time[token] = time.time()

        # 邏輯 1: Normal 語音 (提示類)
        if priority == 2:
            # 如果系統正忙 (不論是 High 或 Normal)，直接丟棄，不排隊
            if is_busy:
                LOGGER.debug(f"語音忙碌，丟棄提示語音: {text}")
                return
            # 如果閒置，放入佇列
            self._enqueue(text, filename, priority, preempt=False, token=token)

        # 邏輯 2: High 語音 (簽到類)
        elif priority == 1:
            if current_p == 2:
                # 正在播 Normal，原本應強制打斷。但為了防止頻繁跳針(震盪)，需檢查間隔
                if (time.time() - self.last_preempt_time) > 1.5:
                    LOGGER.info(f"強制中斷提示語音，插播重要訊息: {text}")
                    self._bump_generation()
                    self._enqueue(text, filename, priority, preempt=True, token=token)
                    self.last_preempt_time = time.time()
                else:
                    LOGGER.info(f"插播過於頻繁，將訊息轉為排隊: {text}")
                    self._enqueue(text, filename, priority, preempt=False, token=token)
            else:
                # 正在播 High 或 閒置 -> 進 Queue 排隊 (High 互不打斷，依序播放)
                # 限制 Queue 長度，避免堆積過多
                if self.queue.qsize() >= 2:
                    try:
                        self.queue.get_nowait() # 丟棄最舊的
                        LOGGER.warning("語音佇列過滿，丟棄舊指令")
                    except:
                        pass
                self._enqueue(text, filename, priority, preempt=False, token=token)

    def _enqueue(self, text, filename, priority, preempt=False, token=None):
        with self.gen_lock:
            gen = self.generation
        self.queue.put((gen, text, filename, priority, preempt, token))

    def _bump_generation(self):
        """內部方法：僅提升版本號，不觸碰 Mixer"""
        with self.gen_lock:
            self.generation += 1

    # 移除公開的 stop() 方法，因為邏輯已整合進 say()
    # def stop(self): ...

    def speak(self):
        """
        語音合成與播放功能主循環。
        """
        # --- 在背景執行緒中執行一次性初始化 ---
        try:
            # 嘗試初始化，若失敗不立即退出，留待迴圈內重試
            # [2026-01-09] 加大 Buffer Size 至 16384 以解決結尾跳幀(回音)問題
            mixer.init(buffer=16384)
            self.mixer_initialized = True
            LOGGER.info("音訊設備在背景執行緒中初始化成功。")
        except Exception as e:
            LOGGER.error(f"初次初始化音訊設備失敗 (將嘗試重試): {e}")

        # --- 主循環 ---
        while not self.stop_threads:
            try:
                # 1. 從佇列取出指令
                gen, text, filename_base, priority, preempt, token = self.queue.get(timeout=0.1)
                
                # [關鍵修正] 如果此指令要求插播 (preempt=True)，則在這裡執行 mixer.stop()
                # 因為這裡是背景執行緒，即使卡住也不會影響主程式
                if preempt:
                    LOGGER.info("執行插播：停止當前播放...")
                    if self.mixer_initialized:
                        try:
                            mixer.music.stop()
                            self.force_reinit = True # 標記重置
                        except Exception as e:
                            LOGGER.error(f"插播停止失敗: {e}")

                # 2. 檢查版本號 (過期指令直接丟棄)
                # 注意：如果是插播指令本身，它的 gen 應該等於 current generation (因為 _bump_generation 先做)
                with self.gen_lock:
                    if gen != self.generation:
                        LOGGER.debug(f"忽略過期語音: {text}")
                        # 雖然過期，但也算結束了，更新狀態以免一直顯示為 playing
                        if token:
                            with self.status_lock:
                                self.last_end_time[token] = time.time()
                        continue

                # 設定當前優先級狀態
                with self.priority_lock:
                    self.current_priority = priority

                LOGGER.info(f"播放語音 (P{priority}): '{text}'")
                filename = filename_base + ".mp3"
                full_path = os.path.join(self.path, filename)
                
                try:
                    # 3. 音訊子系統檢查與重置 (Lazy Re-init)
                    # 只有在未初始化、發生錯誤或被強制要求時才執行 init
                    if not self.mixer_initialized or self.force_reinit:
                        try:
                            if self.mixer_initialized:
                                mixer.quit()
                            # [2026-01-09] 加大 Buffer Size 防止跳幀
                            mixer.init(buffer=16384)
                            self.mixer_initialized = True
                            self.force_reinit = False # 重置成功，清除旗標
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
                    # [2026-01-09] 檢查檔案完整性，防止播放壞檔導致快速迴圈
                    if os.path.getsize(full_path) < 1000: # 小於 1KB 視為異常
                        LOGGER.warning(f"語音檔過小或損毀，跳過播放並刪除: {full_path}")
                        try:
                            os.remove(full_path)
                        except:
                            pass
                        continue

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
                    
                    # 更新 Token 的結束時間
                    if token:
                        with self.status_lock:
                            self.last_end_time[token] = time.time()
                    
                    # [2026-01-09] 強制冷卻一小段時間，讓 ALSA 驅動有時間釋放資源
                    # 這能有效防止連續指令導致的驅動鎖死或聲音卡住
                    time.sleep(0.3)

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