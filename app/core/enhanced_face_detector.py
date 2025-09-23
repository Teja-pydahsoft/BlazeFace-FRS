"""
Enhanced face detector that integrates all improvements
This is the main detector to use in the application
"""

import cv2
import numpy as np
import threading
import time
import queue
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from enum import Enum

@dataclass
class DetectionZone:
    """Detection zone configuration"""
    name: str
    x: int
    y: int
    width: int
    height: int
    min_face_size: int = 30
    max_face_size: int = 200
    confidence_threshold: float = 0.6
    color: Tuple[int, int, int] = (0, 255, 0)

class ProcessingMode(Enum):
    """Processing mode for different performance levels"""
    HIGH_FPS = "high_fps"
    BALANCED = "balanced"
    HIGH_ACCURACY = "high_accuracy"

class EnhancedFaceDetector:
    """Enhanced face detector with all improvements integrated"""
    
    def __init__(self, 
                 processing_mode: ProcessingMode = ProcessingMode.BALANCED,
                 max_workers: int = 2,
                 frame_buffer_size: int = 5):
        """
        Initialize enhanced face detector
        
        Args:
            processing_mode: Processing mode for performance/accuracy tradeoff
            max_workers: Maximum number of worker threads
            frame_buffer_size: Size of frame buffer for processing
        """
        self.processing_mode = processing_mode
        self.max_workers = max_workers
        self.frame_buffer_size = frame_buffer_size
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=frame_buffer_size)
        self.result_queue = queue.Queue(maxsize=frame_buffer_size)
        self.workers = []
        self.running = False
        
        # Detection zones
        self.detection_zones: List[DetectionZone] = []
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Face tracking
        self.face_tracks = {}
        self.next_face_id = 1
        self.frame_count = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection zones
        self._initialize_detection_zones()
    
    def _initialize_detection_zones(self):
        """Initialize detection zones for NVR cameras"""
        # Zone 1: Near field (close to camera)
        self.detection_zones.append(DetectionZone(
            name="Near Field",
            x=0, y=0, width=800, height=400,
            min_face_size=40, max_face_size=150,
            confidence_threshold=0.7,
            color=(0, 255, 0)  # Green
        ))
        
        # Zone 2: Mid field (middle distance)
        self.detection_zones.append(DetectionZone(
            name="Mid Field",
            x=400, y=200, width=800, height=400,
            min_face_size=30, max_face_size=120,
            confidence_threshold=0.6,
            color=(255, 255, 0)  # Yellow
        ))
        
        # Zone 3: Far field (distant)
        self.detection_zones.append(DetectionZone(
            name="Far Field",
            x=800, y=400, width=800, height=400,
            min_face_size=20, max_face_size=80,
            confidence_threshold=0.5,
            color=(255, 165, 0)  # Orange
        ))
        
        # Zone 4: Entry/Exit points
        self.detection_zones.append(DetectionZone(
            name="Entry/Exit",
            x=0, y=400, width=400, height=400,
            min_face_size=25, max_face_size=100,
            confidence_threshold=0.6,
            color=(0, 255, 255)  # Cyan
        ))
    
    def start_workers(self):
        """Start worker threads for processing"""
        if self.running:
            return
        
        self.running = True
        self.workers = []
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} worker threads")
    
    def stop_workers(self):
        """Stop worker threads"""
        self.running = False
        
        # Clear queues to unblock workers
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        self.workers = []
        self.logger.info("Stopped worker threads")
    
    def _worker_loop(self):
        """Worker thread loop for processing frames"""
        from app.core.blazeface_detector import BlazeFaceDetector
        
        # Initialize detector for this worker
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,
            use_opencv_fallback=True
        )
        
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:  # Shutdown signal
                    break
                
                frame, frame_id, timestamp = frame_data
                
                # Process frame
                faces = detector.detect_faces(frame)
                
                # Filter faces by detection zones
                filtered_faces = self._filter_faces_by_zones(frame, faces)
                
                # Add face tracking
                faces_with_ids = self._update_face_tracking(filtered_faces)
                
                # Put result in queue
                result = {
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'faces': faces_with_ids,
                    'fps': self.current_fps
                }
                
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                continue
        
        detector.release()
    
    def _filter_faces_by_zones(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float]]:
        """Filter faces based on detection zones"""
        filtered_faces = []
        
        for face in faces:
            x, y, w, h, conf = face
            center_x, center_y = x + w//2, y + h//2
            
            # Check which zone this face belongs to
            for zone in self.detection_zones:
                if (zone.x <= center_x <= zone.x + zone.width and 
                    zone.y <= center_y <= zone.y + zone.height):
                    
                    # Check zone-specific criteria
                    if (zone.min_face_size <= min(w, h) <= zone.max_face_size and 
                        conf >= zone.confidence_threshold):
                        filtered_faces.append(face)
                        break
        
        return filtered_faces
    
    def _update_face_tracking(self, faces: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float, int]]:
        """Update face tracking and assign IDs"""
        self.frame_count += 1
        faces_with_ids = []
        
        for face in faces:
            x, y, w, h, conf = face
            center_x, center_y = x + w//2, y + h//2
            
            # Find closest existing face
            best_match_id = None
            best_distance = float('inf')
            max_distance = 50.0  # Maximum distance for face tracking
            
            for face_id, track in self.face_tracks.items():
                track_x, track_y, track_w, track_h, track_conf, track_frames = track
                track_center_x = track_x + track_w//2
                track_center_y = track_y + track_h//2
                
                distance = np.sqrt((center_x - track_center_x)**2 + (center_y - track_center_y)**2)
                
                if distance < max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = face_id
            
            if best_match_id is not None:
                # Update existing face
                self.face_tracks[best_match_id] = (x, y, w, h, conf, self.frame_count)
                faces_with_ids.append((x, y, w, h, conf, best_match_id))
            else:
                # Create new face track
                face_id = self.next_face_id
                self.next_face_id += 1
                self.face_tracks[face_id] = (x, y, w, h, conf, self.frame_count)
                faces_with_ids.append((x, y, w, h, conf, face_id))
        
        # Remove old faces (not seen for 10 frames)
        old_faces = [face_id for face_id, track in self.face_tracks.items() 
                    if self.frame_count - track[5] > 10]
        for face_id in old_faces:
            del self.face_tracks[face_id]
        
        return faces_with_ids
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """Add frame to processing queue"""
        if not self.running:
            return False
        
        try:
            frame_id = self.frame_count
            timestamp = time.time()
            
            # Update FPS counter
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:  # Update FPS every 30 frames
                current_time = time.time()
                elapsed = current_time - self.fps_start_time
                self.current_fps = 30 / elapsed
                self.fps_start_time = current_time
            
            self.frame_queue.put_nowait((frame, frame_id, timestamp))
            return True
        except queue.Full:
            return False
    
    def get_result(self) -> Optional[Dict]:
        """Get latest processing result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def draw_detection_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection zones on frame"""
        frame_with_zones = frame.copy()
        
        for zone in self.detection_zones:
            # Draw zone rectangle
            cv2.rectangle(frame_with_zones, 
                         (zone.x, zone.y), 
                         (zone.x + zone.width, zone.y + zone.height),
                         zone.color, 2)
            
            # Draw zone label
            cv2.putText(frame_with_zones, zone.name,
                       (zone.x, zone.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone.color, 2)
        
        return frame_with_zones
    
    def draw_faces_with_ids(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int, float, int]]) -> np.ndarray:
        """Draw faces with IDs and confidence scores"""
        frame_with_faces = frame.copy()
        
        for face in faces:
            x, y, w, h, conf, face_id = face
            
            # Draw face rectangle
            cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw face ID
            cv2.putText(frame_with_faces, f"ID:{face_id}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(frame_with_faces, f"{conf:.2f}",
                       (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame_with_faces
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'current_fps': self.current_fps,
            'frame_count': self.frame_count,
            'active_tracks': len(self.face_tracks),
            'queue_sizes': {
                'frame_queue': self.frame_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            },
            'processing_mode': self.processing_mode.value
        }
