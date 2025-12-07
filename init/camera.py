import queue
import subprocess
import threading
import time
import cv2
import numpy as np
import os
import signal

def _ignore_sigint():
    """Function to be called in the child process to ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class VideoCapture:
    """
    Custom VideoCapture class using FFmpeg to transcode the RTSP stream
    to a pipe of MJPEG images. This is highly robust against codec/stream issues.
    """
    def __init__(self, rtsp_url, config_data=None):
        self.rtsp_url = rtsp_url
        self.q = queue.Queue(maxsize=2)
        self.stop_threads = False
        self.proc = None
        
        self.config = config_data
        
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _log_stderr(self, pipe):
        """Continuously reads from a pipe and logs the output."""
        try:
            for line in iter(pipe.readline, b''):
                print(f'[ffmpeg stderr] {line.decode("utf-8", errors="ignore").strip()}')
        except ValueError:
            # This can happen if the pipe is closed by the main thread during shutdown.
            pass
        finally:
            if not pipe.closed:
                pipe.close()

    def _reader(self):
        """
        The main loop that connects to the stream, pipes MJPEG frames, and decodes them.
        """
        while not self.stop_threads:
            try:
                print(f"Attempting to start FFmpeg MJPEG pipe for {self.rtsp_url}...")
                command = [
                    'ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-rtsp_transport', 'tcp',
                    '-i', self.rtsp_url,
                    '-f', 'image2pipe',
                    '-c:v', 'mjpeg',
                    '-q:v', '2',
                    '-'
                ]
                # Use preexec_fn to make the child process ignore Ctrl+C
                self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=_ignore_sigint)

                stderr_thread = threading.Thread(target=self._log_stderr, args=(self.proc.stderr,))
                stderr_thread.daemon = True
                stderr_thread.start()

                image_buffer = bytearray()
                while not self.stop_threads:
                    chunk = self.proc.stdout.read(4096)
                    if not chunk:
                        print("FFmpeg stdout pipe closed.")
                        break
                    
                    image_buffer.extend(chunk)
                    
                    a = image_buffer.find(b'\xff\xd8')
                    b = image_buffer.find(b'\xff\xd9')

                    if a != -1 and b != -1 and b > a:
                        jpg_data = image_buffer[a:b+2]
                        image_buffer = image_buffer[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            if self.q.full():
                                self.q.get_nowait()
                            self.q.put(frame)
                        else:
                            print("Warning: cv2.imdecode failed to decode frame.")

            except Exception as e:
                print(f"Error in FFmpeg reader thread: {e}")
            finally:
                if self.proc and self.proc.poll() is None:
                    # If the process is still running, it means we exited the loop
                    # due to an error, not a clean shutdown. Terminate it.
                    self.proc.terminate()
                    self.proc.wait(timeout=1) # Give it a moment to die

                # Reconnect logic only runs if it's not a planned shutdown
                if not self.stop_threads:
                    print("Waiting 5 seconds before attempting to reconnect...")
                    time.sleep(5)

    def read(self):
        """
        Retrieves the latest frame from the queue.
        """
        try:
            return self.q.get(timeout=2)
        except queue.Empty:
            print("Frame queue is empty after 2 seconds.")
            return None

    def terminate(self):
        """Stops the reader thread and terminates the FFmpeg subprocess gracefully."""
        print(f"Terminating camera connection to {self.rtsp_url}...")
        self.stop_threads = True
        if self.proc and self.proc.poll() is None:
            print("Sending SIGTERM to FFmpeg process...")
            self.proc.terminate()
        
        # The daemon thread will exit as the main program shuts down.
        # We don't need to join it.
        print(f"Camera connection to {self.rtsp_url} terminated.")
