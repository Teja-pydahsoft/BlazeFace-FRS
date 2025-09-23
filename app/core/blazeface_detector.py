"""
BlazeFace detector implementation using MediaPipe
High-performance face detection optimized for mobile and edge devices
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
import logging

class BlazeFaceDetector:
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_selection: int = 0):
        """
        Initialize BlazeFace detector
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            model_selection: 0 for short-range model, 1 for full-range model
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_selection = model_selection
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        
        self.logger = logging.getLogger(__name__)
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in the given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, width, height, confidence)
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Get confidence score
                    confidence = detection.score[0]
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 0 and height > 0:
                        faces.append((x, y, width, height, confidence))
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {str(e)}")
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
            if hasattr(self, 'face_detection'):
                del self.face_detection
        except Exception as e:
            self.logger.error(f"Error releasing BlazeFace detector: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()
