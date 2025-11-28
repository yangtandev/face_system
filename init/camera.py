import queue
import subprocess
import threading
import time
import cv2
import numpy as np
from urllib.parse import urlparse
import os

class VideoCapture:
    """
    Custom VideoCapture class using a direct FFmpeg subprocess pipe.
    This is the most robust method for handling problematic RTSP streams,
    as it bypasses OpenCV's internal backends (FFmpeg/GStreamer) and uses the
    FFmpeg command-line tool directly.
    """

    def __init__(self, rtsp_url, retries=5, delay=5):
        """
        Initializes the FFmpeg-based video capture object.
        """
        self.rtsp_url = rtsp_url
        self.retries = retries
        self.delay = delay
        
        self.proc = None  # FFmpeg subprocess
        self.width = None
        self.height = None
        
        self.q = queue.Queue(maxsize=2)
        self.stop_threads = False
        
        # Start the reader thread which will manage the connection and frame reading
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _get_video_info(self):
        """
        Uses ffprobe to get the video's width and height.
        """
        print("Probing video stream for resolution...")
        try:
            # Construct the ffprobe command
            command = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                self.rtsp_url
            ]
            
            # Execute the command
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=10)
            
            # Parse the output (e.g., "1920x1080")
            resolution = output.strip().split('x')
            self.width = int(resolution[0])
            self.height = int(resolution[1])
            print(f"Video resolution detected: {self.width}x{self.height}")
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
            print(f"Failed to get video info using ffprobe: {e}")
            # Fallback to a default resolution if ffprobe fails
            self.width, self.height = 1920, 1080
            print(f"Warning: ffprobe failed. Falling back to default resolution {self.width}x{self.height}. This might cause issues.")
            return False # Indicate that we are using a fallback

    def _reader(self):
        """
        The main loop that connects to the stream and reads frames.
        This function runs in a background thread.
        """
        while not self.stop_threads:
            # First, get video info. This is a prerequisite.
            if self.width is None or self.height is None:
                self._get_video_info()

            try:
                print(f"Attempting to start FFmpeg process for RTSP stream at {urlparse(self.rtsp_url).hostname}...")
                
                # Command to decode the RTSP stream and pipe raw BGR frames to stdout
                command = [
                    'ffmpeg',
                    '-hide_banner', '-loglevel', 'error',
                    '-rtsp_transport', 'tcp',      # Force TCP for stability
                    '-i', self.rtsp_url,           # Input URL
                    '-f', 'rawvideo',              # Output format: raw video
                    '-pix_fmt', 'bgr24',           # Pixel format: BGR (what OpenCV uses)
                    '-'                            # Output to stdout
                ]
                
                # Start the FFmpeg subprocess
                self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                frame_size = self.height * self.width * 3
                
                print("FFmpeg process started. Reading frames...")
                
                while not self.stop_threads:
                    # Read a full frame's worth of bytes
                    in_bytes = self.proc.stdout.read(frame_size)
                    
                    if len(in_bytes) == 0:
                        # Stream ended or FFmpeg process died
                        print("FFmpeg stream ended. Attempting to reconnect...")
                        break # Exit inner loop to trigger reconnection
                        
                    if len(in_bytes) != frame_size:
                        print(f"Warning: Incomplete frame received. Expected {frame_size}, got {len(in_bytes)}.")
                        continue

                    # Convert the bytes to a NumPy array and reshape it
                    frame = np.frombuffer(in_bytes, np.uint8).reshape(self.height, self.width, 3)
                    
                    # If the queue is full, discard the oldest frame
                    if self.q.full():
                        self.q.get_nowait()
                        
                    self.q.put(frame)

            except Exception as e:
                print(f"An error occurred in the reader thread: {e}")

            finally:
                # Clean up the subprocess if it exists
                if self.proc:
                    self.proc.kill()
                    self.proc.wait()
                    self.proc = None
                
                # If we are not supposed to stop, wait before retrying
                if not self.stop_threads:
                    print(f"Waiting {self.delay} seconds before reconnecting...")
                    time.sleep(self.delay)

    def read(self):
        """
        Retrieves the latest frame from the queue.
        This will block for up to 1 second for a new frame to become available.
        """
        try:
            # Block for up to 1 second to wait for a new frame
            return self.q.get(timeout=1)
        except queue.Empty:
            # If no frame arrives within the timeout, the stream is likely dead.
            print("Queue empty for 1 second, returning black frame.")
            # Return a black frame of the correct size
            if self.width and self.height:
                 return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else: # Fallback if we don't even know the resolution yet
                 return np.zeros((480, 640, 3), dtype=np.uint8)

    def terminate(self):
        """
        Stops the reader thread and terminates the FFmpeg subprocess.
        """
        print(f"Terminating camera connection to {self.rtsp_url}...")
        self.stop_threads = True
        
        # Kill the FFmpeg process
        if self.proc:
            self.proc.kill()
        
        # Wait for the reader thread to finish
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        print(f"Camera connection to {self.rtsp_url} terminated.")

if __name__ == "__main__":
    # Example usage: Replace with your RTSP URL
    # camera = VideoCapture("rtsp://admin:!QAZ87518499@192.168.31.132:554")
    camera = VideoCapture("rtsp://192.168.50.71") # Using the user's URL

    cv2.namedWindow("FFmpeg Pipe Test", cv2.WINDOW_NORMAL)
    
    while True:
        frame = camera.read()
        
        # We must check if the frame has content, as the black frame fallback is used
        if frame.size > 0:
            # For display, resize the frame to something manageable
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("FFmpeg Pipe Test", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.terminate()
    cv2.destroyAllWindows()