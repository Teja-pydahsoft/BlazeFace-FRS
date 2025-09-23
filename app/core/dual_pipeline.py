"""
Dual Pipeline System for simultaneous human and face detection
Coordinates BlazeFace and Human Detection pipelines
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import List, Tuple, Optional, Dict, Any
import logging

from .blazeface_detector import BlazeFaceDetector
from .human_detector import HumanDetector
from .simple_face_embedder import SimpleFaceEmbedder

class DualPipeline:
    def __init__(self, 
                 config: Dict[str, Any],
                 face_detector: BlazeFaceDetector = None,
                 human_detector: HumanDetector = None,
                 face_embedder: SimpleFaceEmbedder = None):
        """
        Initialize dual pipeline system
        
        Args:
            config: Configuration dictionary
            face_detector: BlazeFace detector instance
            human_detector: Human detector instance
            face_embedder: FaceNet embedder instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.face_detector = face_detector or BlazeFaceDetector(
            min_detection_confidence=config.get('detection_confidence', 0.3)  # Lower threshold for better detection
        )
        self.human_detector = human_detector or HumanDetector(
            confidence_threshold=config.get('detection_confidence', 0.7)
        )
        self.face_embedder = face_embedder or SimpleFaceEmbedder()
        
        # Pipeline state
        self.is_running = False
        self.is_paused = False
        
        # Threading
        self.face_thread = None
        self.human_thread = None
        self.main_thread = None
        
        # Queues for communication
        self.face_queue = queue.Queue(maxsize=10)
        self.human_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Results storage
        self.current_faces = []
        self.current_humans = []
        self.current_embeddings = []
        self.last_update_time = 0
        
        # Synchronization
        self.lock = threading.Lock()
        
    def start_pipeline(self):
        """Start the dual pipeline system"""
        try:
            if self.is_running:
                self.logger.warning("Pipeline is already running")
                return
            
            self.is_running = True
            self.is_paused = False
            
            # Start detection threads
            self.face_thread = threading.Thread(target=self._face_detection_worker, daemon=True)
            self.human_thread = threading.Thread(target=self._human_detection_worker, daemon=True)
            self.main_thread = threading.Thread(target=self._main_processing_worker, daemon=True)
            
            self.face_thread.start()
            self.human_thread.start()
            self.main_thread.start()
            
            self.logger.info("Dual pipeline started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting pipeline: {str(e)}")
            self.stop_pipeline()
    
    def stop_pipeline(self):
        """Stop the dual pipeline system"""
        try:
            self.is_running = False
            self.is_paused = False
            
            # Wait for threads to finish
            if self.face_thread and self.face_thread.is_alive():
                self.face_thread.join(timeout=1.0)
            if self.human_thread and self.human_thread.is_alive():
                self.human_thread.join(timeout=1.0)
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=1.0)
            
            # Clear queues
            self._clear_queues()
            
            self.logger.info("Dual pipeline stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping pipeline: {str(e)}")
    
    def pause_pipeline(self):
        """Pause the pipeline"""
        self.is_paused = True
        self.logger.info("Pipeline paused")
    
    def resume_pipeline(self):
        """Resume the pipeline"""
        self.is_paused = False
        self.logger.info("Pipeline resumed")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through both pipelines
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detection results
        """
        try:
            if not self.is_running or self.is_paused:
                return self._get_empty_result()
            
            # Add frame to processing queues
            if not self.face_queue.full():
                self.face_queue.put(frame.copy())
            if not self.human_queue.full():
                self.human_queue.put(frame.copy())
            
            # Get latest results
            with self.lock:
                result = {
                    'faces': self.current_faces.copy(),
                    'humans': self.current_humans.copy(),
                    'embeddings': self.current_embeddings.copy(),
                    'timestamp': time.time(),
                    'frame_shape': frame.shape
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return self._get_empty_result()
    
    def _face_detection_worker(self):
        """Worker thread for face detection"""
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Get frame from queue
                try:
                    frame = self.face_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                # Debug face detection
                if faces:
                    self.logger.debug(f"Detected {len(faces)} faces")
                    for i, face in enumerate(faces):
                        x, y, w, h, conf = face
                        self.logger.debug(f"  Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                else:
                    self.logger.debug("No faces detected in frame")
                
                # Get face embeddings
                embeddings = []
                for i, face_box in enumerate(faces):
                    x, y, w, h, confidence = face_box
                    face_region = self.face_detector.extract_face_region(frame, (x, y, w, h))
                    if face_region is not None:
                        embedding = self.face_embedder.get_embedding(face_region)
                        if embedding is not None:
                            embeddings.append(embedding)
                            self.logger.debug(f"Face {i}: Got embedding with shape {embedding.shape}")
                        else:
                            self.logger.debug(f"Face {i}: Failed to get embedding")
                    else:
                        self.logger.debug(f"Face {i}: Failed to extract face region")
                
                # Update results
                with self.lock:
                    self.current_faces = faces
                    self.current_embeddings = embeddings
                    self.last_update_time = time.time()
                
                self.face_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in face detection worker: {str(e)}")
                time.sleep(0.1)
    
    def _human_detection_worker(self):
        """Worker thread for human detection"""
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Get frame from queue
                try:
                    frame = self.human_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Detect humans
                humans = self.human_detector.detect_humans(frame)
                
                # Update results
                with self.lock:
                    self.current_humans = humans
                
                self.human_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in human detection worker: {str(e)}")
                time.sleep(0.1)
    
    def _main_processing_worker(self):
        """Main processing worker for coordination"""
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Process any additional coordination logic here
                time.sleep(0.05)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                self.logger.error(f"Error in main processing worker: {str(e)}")
                time.sleep(0.1)
    
    def get_detection_results(self) -> Dict[str, Any]:
        """Get current detection results"""
        with self.lock:
            return {
                'faces': self.current_faces.copy(),
                'humans': self.current_humans.copy(),
                'embeddings': self.current_embeddings.copy(),
                'timestamp': self.last_update_time,
                'is_running': self.is_running,
                'is_paused': self.is_paused
            }
    
    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all detections on the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with all detections drawn
        """
        try:
            result_frame = frame.copy()
            
            # Draw face detections
            if self.current_faces:
                result_frame = self.face_detector.draw_faces(result_frame, self.current_faces)
            
            # Draw human detections
            if self.current_humans:
                result_frame = self.human_detector.draw_humans(result_frame, self.current_humans)
            
            return result_frame
            
        except Exception as e:
            self.logger.error(f"Error drawing detections: {str(e)}")
            return frame
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """Get empty result structure"""
        return {
            'faces': [],
            'humans': [],
            'embeddings': [],
            'timestamp': time.time(),
            'frame_shape': (0, 0, 3)
        }
    
    def _clear_queues(self):
        """Clear all queues"""
        try:
            while not self.face_queue.empty():
                self.face_queue.get_nowait()
            while not self.human_queue.empty():
                self.human_queue.get_nowait()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except:
            pass
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status information"""
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'face_queue_size': self.face_queue.qsize(),
            'human_queue_size': self.human_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'last_update_time': self.last_update_time
        }
    
    def release(self):
        """Release all resources"""
        try:
            self.stop_pipeline()
            
            if self.face_detector:
                self.face_detector.release()
            if self.human_detector:
                self.human_detector.release()
            if self.face_embedder:
                self.face_embedder.release()
                
        except Exception as e:
            self.logger.error(f"Error releasing pipeline: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()
