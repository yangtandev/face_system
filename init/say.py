
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
            # 判斷是否忙碌：正在播 High 或 Normal，或者 Queue 裡還有東西
            is_busy = (self.current_priority != 0) or (not self.queue.empty())
            current_p = self.current_priority

        # 邏輯 1: Normal 語音 (提示類 - 如：請靠近、請抬頭)
        if priority == 2:
            # 嚴格獨佔：只要系統有任何聲音在播 (High 或 Normal) 或在排隊，提示音直接閉嘴
            if is_busy:
                # LOGGER.debug(f"語音忙碌，丟棄提示語音: {text}")
                return
            # 只有完全閒置時才允許播放
            self._enqueue(text, filename, priority, preempt=False, token=token)

        # 邏輯 2: High 語音 (簽到/簽離類)
        elif priority == 1:
            if current_p == 1:
                # 正在播 High：新來的 High 直接丟棄，不排隊！
                # 確保語音頻道絕對乾淨，不會有一堆人名在排隊
                LOGGER.info(f"語音頻道忙碌(High)，丟棄新的重要訊息: {text}")
                return
            
            elif current_p == 2:
                # 正在播 Normal：殺無赦！直接插播。
                # 提示音不重要，簽到音最重要
                LOGGER.info(f"中斷提示語音，插播重要訊息: {text}")
                self._bump_generation() # 提升版本號，讓背景執行緒知道舊指令已無效
                self._enqueue(text, filename, priority, preempt=True, token=token)
            
            else:
                # 閒置或正在排隊 (但 current_p 可能還沒變 1)
                self._enqueue(text, filename, priority, preempt=False, token=token)

    def _enqueue(self, text, filename, priority, preempt=False, token=None):
        # [2026-01-10 關鍵修正] 只有在真正入隊時才記錄開始時間
        # 防止「指令被丟棄但 ID 卻被鎖死在正在播放狀態」的 Bug
        if token:
            with self.status_lock:
                self.last_start_time[token] = time.time()
                
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
            # [2026-01-10] 修正：降低 Buffer Size 至 4096 以避免 underrun 與延遲
            mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=4096)
            try:
                mixer.init()
            except Exception:
                LOGGER.warning("無法初始化音訊設備，切換至 Dummy Driver (無聲模式)...")
                os.environ['SDL_AUDIODRIVER'] = 'dummy' # 使用虛擬音效卡
                mixer.init(buffer=4096)

            self.mixer_initialized = True
            LOGGER.info("音訊設備在背景執行緒中初始化成功。")
        except Exception as e:
            LOGGER.error(f"音訊設備初始化完全失敗，語音功能將停用: {e}")
            self.mixer_initialized = False

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
                            if mixer.music.get_busy():
                                mixer.music.stop()
                                time.sleep(0.1)
                        except Exception as e:
                            LOGGER.error(f"插播停止失敗: {e}")
                            self.mixer_initialized = False

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
                    # 3. 音訊子系統檢查
                    # 僅在尚未初始化(或上次失敗)時嘗試
                    if not self.mixer_initialized:
                        try:
                            # 嘗試重新初始化
                            mixer.init()
                            self.mixer_initialized = True
                            LOGGER.info("音訊子系統重新初始化成功。")
                        except Exception as e:
                            LOGGER.error(f"音訊初始化失敗: {e}")
                            # 稍後重試
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
                    while True:
                        try:
                            if not mixer.music.get_busy():
                                break
                        except:
                            break
                        
                        if time.time() - start_wait > 10:
                            break

                        # 檢查是否被插播 (stop 呼叫)
                        with self.gen_lock:
                            if gen != self.generation:
                                try:
                                    mixer.music.stop()
                                except:
                                    pass
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