"""
OpenCV-based face detector as a reliable alternative to MediaPipe
Provides consistent face detection for NVR cameras
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

class OpenCVFaceDetector:
    """OpenCV-based face detector optimized for NVR cameras"""
    
    def __init__(self, 
                 scale_factor: float = 1.05,
                 min_neighbors: int = 4,
                 min_size: Tuple[int, int] = (25, 25),
                 max_size: Tuple[int, int] = (250, 250)):
        """
        Initialize OpenCV face detector
        
        Args:
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible object size. Objects smaller than this are ignored
            max_size: Maximum possible object size. Objects larger than this are ignored
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.max_size = max_size
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV face cascade")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("OpenCV face detector initialized")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in the given frame
        
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
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                maxSize=self.max_size
            )
            
            # Filter and improve detections with much stricter criteria
            face_list = []
            for (x, y, w, h) in faces:
                # Basic size validation - faces should be reasonable size
                if w < 30 or h < 30 or w > 200 or h > 200:
                    continue
                
                # Check aspect ratio (faces should be roughly square-ish)
                aspect_ratio = w / h
                if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                    continue
                
                # Extract face region for validation
                face_region = gray[y:y+h, x:x+w]
                if face_region.size == 0:
                    continue
                
                # Check if region has enough variation (not just a solid color)
                face_std = np.std(face_region)
                if face_std < 15:  # Too uniform, likely not a face
                    continue
                
                # Check for face-like features using edge detection
                edges = cv2.Canny(face_region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                if edge_density < 0.03:  # Not enough edges for a face (reduced threshold)
                    continue
                
                # Check position - faces should be in upper 3/4 of frame
                center_x, center_y = x + w//2, y + h//2
                if center_y > frame.shape[0] * 0.8:  # Too low in frame (more lenient)
                    continue
                
                # Check for skin-like colors (basic check)
                face_bgr = frame[y:y+h, x:x+w]
                if len(face_bgr.shape) == 3:
                    # Convert to HSV for better skin detection
                    face_hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
                    # Define skin color range in HSV
                    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                    skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
                    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
                    
                    # If less than 5% skin-like pixels, likely not a face (more lenient)
                    if skin_ratio < 0.05:
                        continue
                
                # Calculate confidence based on multiple factors
                frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
                
                # Distance from center (normalized)
                distance_from_center = np.sqrt(
                    ((center_x - frame_center_x) / frame_center_x) ** 2 + 
                    ((center_y - frame_center_y) / frame_center_y) ** 2
                )
                
                # Face size factor (reasonable size faces get higher confidence)
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                size_factor = min(face_area / (frame_area * 0.02), 1.0)  # 2% of frame area max
                
                # Position factor (faces in upper 2/3 of frame are more likely)
                position_factor = 1.0 if center_y < frame.shape[0] * 0.67 else 0.5
                
                # Variation factor (more variation = more likely to be a face)
                variation_factor = min(face_std / 40.0, 1.0)
                
                # Edge factor (more edges = more likely to be a face)
                edge_factor = min(edge_density * 10, 1.0)
                
                # Skin factor (more skin-like pixels = more likely to be a face)
                skin_factor = min(skin_ratio * 2, 1.0) if len(face_bgr.shape) == 3 else 0.5
                
                # Calculate confidence (0.4 to 0.95 range) - much stricter
                confidence = 0.4 + (size_factor * 0.2) + (variation_factor * 0.15) + (position_factor * 0.1) + (edge_factor * 0.1) + (skin_factor * 0.15) - (distance_from_center * 0.2)
                confidence = max(0.4, min(0.95, confidence))
                
                # Only keep detections with reasonable confidence
                if confidence > 0.5:
                    face_list.append((x, y, w, h, confidence))
            
            # Remove overlapping detections (non-maximum suppression)
            face_list = self._remove_overlapping_detections(face_list)
            
            if face_list:
                self.logger.debug(f"Detected {len(face_list)} faces")
            else:
                self.logger.debug("No faces detected")
            
            return face_list
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def _remove_overlapping_detections(self, faces: List[Tuple[int, int, int, int, float]], 
                                     overlap_threshold: float = 0.2) -> List[Tuple[int, int, int, int, float]]:
        """
        Remove overlapping face detections using non-maximum suppression
        
        Args:
            faces: List of face detections (x, y, w, h, confidence)
            overlap_threshold: IoU threshold for considering detections as overlapping
            
        Returns:
            Filtered list of non-overlapping face detections
        """
        if len(faces) <= 1:
            return faces
        
        # Sort by confidence (highest first)
        faces_sorted = sorted(faces, key=lambda x: x[4], reverse=True)
        
        keep = []
        while faces_sorted:
            # Take the detection with highest confidence
            current = faces_sorted.pop(0)
            keep.append(current)
            
            # Remove all detections that overlap significantly with current
            remaining = []
            for face in faces_sorted:
                if self._calculate_iou(current, face) < overlap_threshold:
                    remaining.append(face)
            
            faces_sorted = remaining
        
        return keep
    
    def _calculate_iou(self, face1: Tuple[int, int, int, int, float], 
                      face2: Tuple[int, int, int, int, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two face detections
        
        Args:
            face1: First face detection (x, y, w, h, confidence)
            face2: Second face detection (x, y, w, h, confidence)
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1, _ = face1
        x2, y2, w2, h2, _ = face2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
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
        Get face landmarks using OpenCV (simplified version)
        Note: OpenCV doesn't provide detailed landmarks like MediaPipe
        This returns basic face center points
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face landmarks (simplified)
        """
        try:
            faces = self.detect_faces(frame)
            landmarks = []
            
            for (x, y, w, h, _) in faces:
                # Create basic landmarks (face center, eye positions, etc.)
                face_landmarks = [
                    [x + w//2, y + h//2],  # Face center
                    [x + w//3, y + h//3],  # Left eye (approximate)
                    [x + 2*w//3, y + h//3],  # Right eye (approximate)
                    [x + w//2, y + 2*h//3],  # Nose (approximate)
                    [x + w//3, y + 3*h//4],  # Left mouth corner (approximate)
                    [x + 2*w//3, y + 3*h//4]  # Right mouth corner (approximate)
                ]
                landmarks.append(np.array(face_landmarks))
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Error getting face landmarks: {str(e)}")
            return []
    
    def release(self):
        """Release resources"""
        try:
            # OpenCV doesn't need explicit cleanup
            pass
        except Exception as e:
            self.logger.error(f"Error releasing OpenCV face detector: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()
