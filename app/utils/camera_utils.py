"""
Camera utilities for BlazeFace-FRS system
Handles camera initialization, frame capture, and camera management
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
import threading
import time

class CameraManager:
    def __init__(self, camera_source: str = "0", width: int = 640, height: int = 480):
        """
        Initialize camera manager
        
        Args:
            camera_source: Camera source - can be:
                - "0" or 0: Default webcam
                - "rtsp://ip:port/path": RTSP stream (NVR camera)
                - "http://ip:port/video": HTTP stream
                - "/path/to/video.mp4": Video file
            width: Frame width
            height: Frame height
        """
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        # Camera state
        self.is_initialized = False
        self.is_capturing = False
        self.camera_type = self._determine_camera_type(camera_source)
        
        # Threading
        self.capture_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Initialize camera
        self._init_camera()
    
    def _determine_camera_type(self, source):
        """Determine camera type from source"""
        if isinstance(source, int) or source.isdigit():
            return "webcam"
        elif source.startswith(("rtsp://", "http://", "https://")):
            return "stream"
        elif source.endswith((".mp4", ".avi", ".mov", ".mkv")):
            return "video_file"
        else:
            return "unknown"
    
    def _init_camera(self):
        """Initialize camera"""
        try:
            # Add small delay to prevent camera conflicts
            time.sleep(0.1)
            
            # Handle different camera types
            if self.camera_type == "webcam":
                camera_index = int(self.camera_source) if isinstance(self.camera_source, str) else self.camera_source
                self.cap = cv2.VideoCapture(camera_index)
            else:
                # For streams and video files, use the source directly
                self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera source: {self.camera_source}")
                return False
            
            # Set camera properties for different types
            if self.camera_type == "webcam":
                # Set camera properties for webcam
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increased buffer for stability
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for better performance
            elif self.camera_type == "stream":
                # For RTSP/HTTP streams, set buffer size and timeout
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Set timeout for stream connections (in milliseconds)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # Reduced timeout for faster recovery
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # Reduced timeout for faster recovery
                # Don't force codec conversion - let OpenCV handle the original codec
                # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                # Try to set frame rate
                self.cap.set(cv2.CAP_PROP_FPS, 25)  # NVR cameras often use 25fps
            
            # Verify camera settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            self.is_initialized = False
            return False
    
    def is_camera_available(self) -> bool:
        """Check if camera is available and working"""
        try:
            if not self.is_initialized or self.cap is None:
                return False
            
            # For streams, be more lenient with availability check
            if self.camera_type == "stream":
                # Just check if camera is opened, don't try to read frame
                return self.cap.isOpened()
            else:
                # For webcams, check if camera is opened first
                if not self.cap.isOpened():
                    return False
                
                # Try to read a frame but don't fail immediately
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True
                # If frame read fails, still return True if camera is opened
                # This prevents false negatives due to temporary frame read issues
                return True
            
        except Exception as e:
            self.logger.error(f"Error checking camera availability: {str(e)}")
            return False
    
    def is_camera_in_use(self) -> bool:
        """Check if camera is currently being used by this manager"""
        return self.is_capturing
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get a single frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        try:
            if not self.is_initialized or self.cap is None:
                return False, None
            
            # Try to read frame with retry logic for streams
            ret, frame = self.cap.read()
            
            # For streams, if first read fails, try a few more times
            if not ret and self.camera_type == "stream":
                for attempt in range(3):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.1)  # Small delay between attempts
            
            if ret and frame is not None:
                # For NVR streams, validate frame quality
                if self.camera_type == "stream":
                    # Check if frame is not corrupted
                    if frame.size == 0 or np.all(frame == 0):
                        self.logger.warning("Received empty or corrupted frame from stream")
                        return False, None
                    
                    # Check frame dimensions
                    if len(frame.shape) != 3 or frame.shape[2] != 3:
                        self.logger.warning(f"Invalid frame shape from stream: {frame.shape}")
                        return False, None
                
                # For NVR streams, preserve original resolution for better face detection
                # Only resize if explicitly requested and different from original
                if self.camera_type == "stream":
                    # Keep original resolution for face detection
                    self.logger.debug(f"Captured frame: {frame.shape}")
                    return True, frame
                else:
                    # For webcams, resize to requested dimensions
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    # Log frame info for debugging
                    self.logger.debug(f"Captured frame: {frame.shape}")
                    
                    return True, frame
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error getting frame: {str(e)}")
            return False, None
    
    def start_capture(self):
        """Start continuous frame capture in a separate thread"""
        try:
            if self.is_capturing:
                return
            
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("Camera capture started")
            
        except Exception as e:
            self.logger.error(f"Error starting capture: {str(e)}")
    
    def stop_capture(self):
        """Stop continuous frame capture"""
        try:
            self.is_capturing = False
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            
            self.logger.info("Camera capture stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping capture: {str(e)}")
    
    def _capture_worker(self):
        """Worker thread for continuous frame capture"""
        while self.is_capturing:
            try:
                ret, frame = self.get_frame()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                else:
                    time.sleep(0.01)  # Small delay if no frame available
                    
            except Exception as e:
                self.logger.error(f"Error in capture worker: {str(e)}")
                time.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame
        
        Returns:
            Latest frame or None if not available
        """
        try:
            with self.frame_lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting latest frame: {str(e)}")
            return None
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_initialized or self.cap is None:
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify the change
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                self.width = width
                self.height = height
                self.logger.info(f"Camera resolution set to {width}x{height}")
                return True
            else:
                self.logger.warning(f"Failed to set resolution to {width}x{height}, got {actual_width}x{actual_height}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            return False
    
    def set_fps(self, fps: int) -> bool:
        """
        Set camera FPS
        
        Args:
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_initialized or self.cap is None:
                return False
            
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Verify the change
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera FPS set to {fps} (actual: {actual_fps})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting FPS: {str(e)}")
            return False
    
    def get_camera_info(self) -> dict:
        """
        Get camera information
        
        Returns:
            Dictionary containing camera properties
        """
        try:
            if not self.is_initialized or self.cap is None:
                return {}
            
            info = {
                'camera_source': self.camera_source,
                'camera_type': self.camera_type,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'is_opened': self.cap.isOpened()
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting camera info: {str(e)}")
            return {}
    
    def list_available_cameras(self) -> List[int]:
        """
        List available camera indices
        
        Returns:
            List of available camera indices
        """
        available_cameras = []
        
        try:
            # Test cameras 0-9
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                    cap.release()
                    
        except Exception as e:
            self.logger.error(f"Error listing cameras: {str(e)}")
        
        return available_cameras
    
    def capture_image(self, filepath: str) -> bool:
        """
        Capture and save a single image
        
        Args:
            filepath: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            ret, frame = self.get_frame()
            if ret and frame is not None:
                cv2.imwrite(filepath, frame)
                self.logger.info(f"Image saved to {filepath}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error capturing image: {str(e)}")
            return False
    
    def release(self):
        """Release camera resources"""
        try:
            self.stop_capture()
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.is_initialized = False
            self.logger.info("Camera released")
            
            # Add small delay to prevent rapid reinitialization
            time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"Error releasing camera: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()
