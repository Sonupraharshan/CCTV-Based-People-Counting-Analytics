"""
Video Reader Utility
Async video and RTSP stream reader with frame buffering.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import threading
from queue import Queue, Empty
import time


class VideoReader:
    """
    Asynchronous video reader for files and RTSP streams.
    
    Uses a separate thread to read frames ahead of time,
    improving performance and reducing latency.
    """
    
    def __init__(
        self,
        source: str,
        buffer_size: int = 30,
        resize: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize video reader.
        
        Args:
            source: Video file path or RTSP URL
            buffer_size: Number of frames to buffer
            resize: Optional (width, height) to resize frames
        """
        self.source = source
        self.buffer_size = buffer_size
        self.resize = resize
        
        # Open video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Frame buffer
        self.frame_queue = Queue(maxsize=buffer_size)
        
        # Threading
        self.thread = None
        self.running = False
        self.frame_count = 0
        
        # Check if source is RTSP
        self.is_rtsp = source.startswith('rtsp://')
        
        print(f"VideoReader initialized:")
        print(f"  Source: {source}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        if not self.is_rtsp:
            print(f"  Total frames: {self.total_frames}")
    
    def start(self):
        """Start async frame reading."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        print("VideoReader started")
    
    def _reader_thread(self):
        """Background thread to read frames."""
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                
                if not ret:
                    # End of video or read error
                    if self.is_rtsp:
                        # Try to reconnect for RTSP
                        print("RTSP connection lost, attempting reconnect...")
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(self.source)
                        if not self.cap.isOpened():
                            print("Failed to reconnect")
                            self.running = False
                            break
                        continue
                    else:
                        # End of file
                        self.running = False
                        break
                
                # Resize if needed
                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize)
                
                # Add to queue
                try:
                    self.frame_queue.put((self.frame_count, frame), timeout=1)
                    self.frame_count += 1
                except:
                    pass
            else:
                # Buffer full, wait a bit
                time.sleep(0.001)
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Read next frame from buffer.
        
        Args:
            timeout: Maximum time to wait for frame
        
        Returns:
            Tuple of (success, frame, frame_number)
        """
        try:
            frame_num, frame = self.frame_queue.get(timeout=timeout)
            return True, frame, frame_num
        except Empty:
            return False, None, -1
    
    def is_opened(self) -> bool:
        """Check if video source is open."""
        return self.running or not self.frame_queue.empty()
    
    def stop(self):
        """Stop async reading and release resources."""
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=2)
        
        self.cap.release()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        print("VideoReader stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class VideoWriter:
    """Simple video writer wrapper."""
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            frame_size: (width, height) of frames
            codec: FourCC codec code
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f" Failed to create video writer: {output_path}")
        
        print(f"VideoWriter created: {output_path} ({frame_size[0]}x{frame_size[1]} @ {fps}fps)")
    
    def write(self, frame: np.ndarray):
        """Write frame to video."""
        self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        self.writer.release()
        print(f"Video saved: {self.output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
