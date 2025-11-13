import subprocess
import time
import sys
import os

MEDIAMTX_PATH = 'mediamtx/mediamtx.exe'
FFMPEG_COMMAND = [
    'ffmpeg',
    '-f', 'dshow',
    '-i', 'video=USB2.0 HD UVC WebCam',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-f', 'rtsp',
    'rtsp://localhost:8554/stream'
]
RTSP_STREAM_NAME = '/mystream'

def main():
    mediamtx_process = None
    ffmpeg_process = None

    try:
        # 啟動 mediamtx 伺服器
        print(f"1. 正在啟動 mediamtx 伺服器...")
        # 使用 DEVNULL 將子程序的輸出隱藏，保持主控台乾淨
        mediamtx_process = subprocess.Popen(
            [MEDIAMTX_PATH],
            cwd=os.path.dirname('mediamtx/mediamtx.exe'),
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        )
        print("   mediamtx 伺服器已在背景執行。")
        time.sleep(2) # 等待伺服器完全啟動

        # 啟動 FFmpeg 推流
        print(f"2. 正在啟動 FFmpeg 將攝影機畫面推流...")
        ffmpeg_process = subprocess.Popen(
            FFMPEG_COMMAND,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        )
        print("   FFmpeg 已開始推流。")
        print("-" * 30)

        # 顯示連線資訊
        try:
            import socket
            print(f"請在其他程式中使用以下 RTSP 網址連線：")
            print(f"   rtsp://localhost:8554{RTSP_STREAM_NAME}")
        except ImportError:
            print("請查詢你的筆電 IP 位址，並用 rtsp://<你的IP>:8554/mystream 來連線。")

        print("\n程式正在運行中，請保持此視窗開啟。")
        print("若要停止，請在此視窗按下 Ctrl + C。")

        # 保持主程式運行，直到使用者中斷
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # 當使用者按下 Ctrl+C 時，會觸發此處的程式碼
        print("\n偵測到使用者中斷 (Ctrl+C)，正在關閉所有程式...")
    except Exception as e:
        print(f"\n發生未預期的錯誤: {e}")
    finally:
        # 無論程式是正常結束還是發生錯誤，這個區塊都會被執行
        # 確保子程式被確實關閉
        if ffmpeg_process:
            print("   正在停止 FFmpeg...")
            ffmpeg_process.terminate() # 傳送終止訊號
            ffmpeg_process.wait()      # 等待程式完全結束

        if mediamtx_process:
            print("   正在停止 mediamtx 伺服器...")
            mediamtx_process.terminate()
            mediamtx_process.wait()

        print("所有程式均已安全關閉。")

if __name__ == '__main__':
    main()
