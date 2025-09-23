"""
BlazeFace detector implementation using MediaPipe with OpenCV fallback
High-performance face detection optimized for mobile and edge devices
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
import logging

# Import OpenCV fallback
try:
    from .opencv_face_detector import OpenCVFaceDetector
except ImportError:
    from opencv_face_detector import OpenCVFaceDetector

class BlazeFaceDetector:
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_selection: int = 0,
                 use_opencv_fallback: bool = True):
        """
        Initialize BlazeFace detector with MediaPipe and OpenCV fallback
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            model_selection: 0 for short-range model, 1 for full-range model
            use_opencv_fallback: Whether to use OpenCV as fallback when MediaPipe fails
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_selection = model_selection
        self.use_opencv_fallback = use_opencv_fallback
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        try:
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
            self.mediapipe_available = True
            self.logger.info("MediaPipe face detection initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MediaPipe: {e}")
            self.mediapipe_available = False
            self.face_detection = None
        
        # Initialize OpenCV fallback
        if use_opencv_fallback:
            try:
                self.opencv_detector = OpenCVFaceDetector()
                self.opencv_available = True
                self.logger.info("OpenCV face detection fallback initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenCV fallback: {e}")
                self.opencv_available = False
                self.opencv_detector = None
        else:
            self.opencv_available = False
            self.opencv_detector = None
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in the given frame using MediaPipe with OpenCV fallback
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, width, height, confidence)
        """
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                self.logger.warning("Empty or invalid frame provided")
                return []
            
            # Ensure frame has 3 channels
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.logger.warning(f"Invalid frame shape: {frame.shape}")
                return []
            
            # Log frame info for debugging
            h, w, c = frame.shape
            self.logger.debug(f"Processing frame: {w}x{h}x{c}")
            
            # Try MediaPipe first if available
            if self.mediapipe_available and self.face_detection is not None:
                faces = self._detect_faces_mediapipe(frame)
                if faces:
                    self.logger.debug(f"MediaPipe detected {len(faces)} faces")
                    return faces
                else:
                    self.logger.debug("MediaPipe found no faces, trying OpenCV fallback")
            
            # Use OpenCV fallback if MediaPipe failed or is not available
            if self.opencv_available and self.opencv_detector is not None:
                faces = self.opencv_detector.detect_faces(frame)
                if faces:
                    self.logger.debug(f"OpenCV fallback detected {len(faces)} faces")
                    return faces
                else:
                    self.logger.debug("OpenCV fallback found no faces")
            
            self.logger.debug("No faces detected by any method")
            return []
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def _detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using MediaPipe"""
        try:
            h, w, c = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # For NVR cameras, try different preprocessing approaches
            detection_attempts = []
            
            # 1. Original frame (for high-res NVR streams)
            detection_attempts.append(rgb_frame)
            
            # 2. Resized to optimal face detection size (MediaPipe works best with 640x480)
            if (w, h) != (640, 480):
                detection_attempts.append(cv2.resize(rgb_frame, (640, 480)))
            
            # 3. Resized to 1280x720 (common NVR resolution)
            if (w, h) != (1280, 720):
                detection_attempts.append(cv2.resize(rgb_frame, (1280, 720)))
            
            # 4. Enhanced contrast for better detection in low light
            detection_attempts.append(cv2.convertScaleAbs(rgb_frame, alpha=1.3, beta=15))
            
            # 5. Histogram equalization for better contrast
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            detection_attempts.append(cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB))
            
            # 6. Gaussian blur + sharpening for noise reduction
            blurred = cv2.GaussianBlur(rgb_frame, (3, 3), 0)
            sharpened = cv2.addWeighted(rgb_frame, 1.5, blurred, -0.5, 0)
            detection_attempts.append(sharpened)
            
            # 7. For very high resolution NVR streams, try a smaller version
            if w > 1920:  # If width > 1920, try a smaller version
                smaller = cv2.resize(rgb_frame, (1280, 720))
                detection_attempts.append(smaller)
            
            best_faces = []
            best_confidence = 0
            
            for i, processed_frame in enumerate(detection_attempts):
                try:
                    # Process the frame
                    results = self.face_detection.process(processed_frame)
                    
                    if results.detections:
                        faces = []
                        current_confidence = 0
                        
                        for detection in results.detections:
                            # Get bounding box
                            bbox = detection.location_data.relative_bounding_box
                            
                            # Get confidence score
                            confidence = detection.score[0]
                            current_confidence += confidence
                            
                            # Get processed frame dimensions
                            proc_h, proc_w = processed_frame.shape[:2]
                            
                            # Convert relative coordinates to absolute based on processed frame
                            x_proc = int(bbox.xmin * proc_w)
                            y_proc = int(bbox.ymin * proc_h)
                            width_proc = int(bbox.width * proc_w)
                            height_proc = int(bbox.height * proc_h)
                            
                            # Scale coordinates back to original frame dimensions
                            scale_x = w / proc_w
                            scale_y = h / proc_h
                            
                            x = int(x_proc * scale_x)
                            y = int(y_proc * scale_y)
                            width = int(width_proc * scale_x)
                            height = int(height_proc * scale_y)
                            
                            # Ensure coordinates are within frame bounds
                            x = max(0, min(x, w - 1))
                            y = max(0, min(y, h - 1))
                            width = min(width, w - x)
                            height = min(height, h - y)
                            
                            if width > 0 and height > 0:
                                faces.append((x, y, width, height, confidence))
                        
                        # Keep the best detection result
                        if faces and current_confidence > best_confidence:
                            best_faces = faces
                            best_confidence = current_confidence
                            self.logger.debug(f"MediaPipe attempt {i+1}: Found {len(faces)} faces with avg confidence {current_confidence/len(faces):.3f}")
                
                except Exception as e:
                    self.logger.debug(f"MediaPipe attempt {i+1} failed: {str(e)}")
                    continue
            
            return best_faces
            
        except Exception as e:
            self.logger.error(f"Error in MediaPipe face detection: {str(e)}")
            return []
    
    def draw_faces(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw face bounding boxes on the frame
        
        Args:
            frame: Input image frame
            faces: List of face bounding boxes
            
        Returns:
            Frame with drawn face bounding boxes
        """
        try:
            for (x, y, w, h, confidence) in faces:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence score
                label = f"Face: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing faces: {str(e)}")
            return frame
    
    def extract_face_region(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face region from frame
        
        Args:
            frame: Input image frame
            face_box: Face bounding box (x, y, width, height)
            
        Returns:
            Extracted face region or None if extraction fails
        """
        try:
            x, y, w, h = face_box
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size > 0:
                return face_region
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting face region: {str(e)}")
            return None
    
    def get_face_landmarks(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Get face landmarks using MediaPipe
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face landmarks
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            landmarks = []
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    if detection.location_data.relative_keypoints:
                        face_landmarks = []
                        for keypoint in detection.location_data.relative_keypoints:
                            x = int(keypoint.x * w)
                            y = int(keypoint.y * h)
                            face_landmarks.append([x, y])
                        landmarks.append(np.array(face_landmarks))
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Error getting face landmarks: {str(e)}")
            return []
    
    def release(self):
        """Release resources"""
        try:
            if hasattr(self, 'face_detection') and self.face_detection is not None:
                del self.face_detection
            
            if hasattr(self, 'opencv_detector') and self.opencv_detector is not None:
                self.opencv_detector.release()
        except Exception as e:
            self.logger.error(f"Error releasing BlazeFace detector: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()
