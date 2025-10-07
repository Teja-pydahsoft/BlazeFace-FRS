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
import os

from .blazeface_detector import BlazeFaceDetector
from .human_detector import HumanDetector
from .standard_face_embedder import StandardFaceEmbedder
from .simple_face_embedder import SimpleFaceEmbedder
from .insightface_embedder import InsightFaceEmbedder
from .faiss_index import FaissIndex

class DualPipeline:
    def __init__(self, 
                 config: Dict[str, Any],
                 face_detector: BlazeFaceDetector = None,
                 human_detector: HumanDetector = None,
                 face_embedder: StandardFaceEmbedder = None):
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
        # Use very low confidence threshold for NVR cameras
        detection_conf = config.get('detection_confidence', 0.1) if config.get('camera_type') == 'stream' else config.get('detection_confidence', 0.3)
        self.face_detector = face_detector or BlazeFaceDetector(
            min_detection_confidence=detection_conf
        )
        self.human_detector = human_detector or HumanDetector(
            confidence_threshold=config.get('detection_confidence', 0.7)
        )
        # Initialize face embedder to match database encodings
        if face_embedder is not None:
            self.face_embedder = face_embedder
        else:
            # Prefer InsightFace (buffalo_l) 512-d embeddings as canonical embedder
            try:
                self.face_embedder = InsightFaceEmbedder(model_name='buffalo_l')
                self.logger.info("Using InsightFaceEmbedder (buffalo_l, 512-d) for pipeline")
            except Exception as e:
                self.logger.warning(f"InsightFaceEmbedder init failed, falling back to other embedders: {e}")
                try:
                    self.face_embedder = StandardFaceEmbedder(model='large')  # 128-dimensional embeddings
                    self.logger.info("Using StandardFaceEmbedder (128-d) as fallback")
                except Exception:
                    self.face_embedder = SimpleFaceEmbedder()
                    self.logger.info("Using SimpleFaceEmbedder as final fallback")
        
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
        # Attempt to load FAISS index if present
        try:
            self.faiss_index = FaissIndex()
            self.faiss_index.load(os.path.join('data', 'faiss'))
            if self.faiss_index.dim is None:
                self.faiss_index = None
            else:
                self.logger.info('Loaded FAISS index for fast recognition')
        except Exception as e:
            self.faiss_index = None
            self.logger.debug(f'No FAISS index loaded: {e}')
        
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
        Process a single frame through both pipelines with performance optimizations
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detection results
        """
        try:
            if not self.is_running or self.is_paused:
                return self._get_empty_result()
            
            # Performance optimization: Skip processing if too frequent
            current_time = time.time()
            if not hasattr(self, '_last_process_time'):
                self._last_process_time = 0
            
            # Limit processing frequency for better performance
            max_fps = self.config.get('performance_settings', {}).get('max_processing_fps', 15)
            min_interval = 1.0 / max_fps if max_fps > 0 else 0.1
            
            if current_time - self._last_process_time < min_interval:
                # Return cached results if processing too frequently
                with self.lock:
                    return {
                        'faces': self.current_faces.copy(),
                        'humans': self.current_humans.copy(),
                        'embeddings': self.current_embeddings.copy(),
                        'timestamp': current_time,
                        'frame_shape': frame.shape
                    }
            
            self._last_process_time = current_time
            
            # For NVR cameras and streams, use direct processing for better reliability
            if hasattr(self, 'config') and self.config.get('camera_type') == 'stream':
                return self._process_frame_direct(frame)
            
            # Add frame to processing queues for webcam (with size limit)
            if not self.face_queue.full():
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (640, 480)) if frame.shape[1] > 640 else frame
                self.face_queue.put(small_frame.copy())
            if not self.human_queue.full():
                self.human_queue.put(frame.copy())
            
            # Get latest results
            with self.lock:
                result = {
                    'faces': self.current_faces.copy(),
                    'humans': self.current_humans.copy(),
                    'embeddings': self.current_embeddings.copy(),
                    'timestamp': current_time,
                    'frame_shape': frame.shape
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return self._get_empty_result()
    
    def _process_frame_direct(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process frame directly without threading (for NVR cameras)
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Resize frame for consistent processing, especially for high-res NVR streams
            nvr_settings = self.config.get('nvr_settings', {})
            target_width = nvr_settings.get('frame_width', 640)
            target_height = nvr_settings.get('frame_height', 480)
            
            processing_frame = cv2.resize(frame, (target_width, target_height))
            
            # Detect faces directly
            faces = self.face_detector.detect_faces(processing_frame)
            
            # Debug face detection
            if faces:
                self.logger.debug(f"Direct detection: Found {len(faces)} faces")
                for i, face in enumerate(faces):
                    x, y, w, h, conf = face
                    self.logger.debug(f"  Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
            else:
                self.logger.debug("Direct detection: No faces found")
            
            # Get face embeddings
            # Run InsightFace once on the full processing frame and map embeddings to detected boxes via IOU
            embeddings = [None] * len(faces)
            try:
                # InsightFace expects RGB images in our codepaths; ensure conversion
                proc_for_insight = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB) if processing_frame.shape[2] == 3 else processing_frame
                insight_results = self.face_embedder.detect_and_encode_faces(proc_for_insight)

                # Convert insightface bboxes to [x1,y1,x2,y2] format and compute IOU matching
                def to_xyxy(bbox):
                    x1, y1, x2, y2 = bbox
                    return [int(x1), int(y1), int(x2), int(y2)]

                def iou(boxA, boxB):
                    # boxA: x,y,w,h ; boxB: x1,y1,x2,y2
                    ax, ay, aw, ah = boxA
                    a1x, a1y, a2x, a2y = ax, ay, ax+aw, ay+ah
                    bx1, by1, bx2, by2 = boxB
                    xA = max(a1x, bx1)
                    yA = max(a1y, by1)
                    xB = min(a2x, bx2)
                    yB = min(a2y, by2)
                    interW = max(0, xB - xA)
                    interH = max(0, yB - yA)
                    interArea = interW * interH
                    boxAArea = (a2x - a1x) * (a2y - a1y)
                    boxBArea = (bx2 - bx1) * (by2 - by1)
                    denom = float(boxAArea + boxBArea - interArea + 1e-6)
                    return interArea / denom if denom > 0 else 0.0

                used = set()
                for i, face_box in enumerate(faces):
                    best_iou = 0.0
                    best_emb = None
                    bx, by, bw, bh, conf = face_box
                    for j, (face_info, embedding) in enumerate(insight_results):
                        if embedding is None:
                            continue
                        b_xyxy = to_xyxy(face_info['bbox'])
                        score = iou((bx, by, bw, bh), b_xyxy)
                        if score > best_iou:
                            best_iou = score
                            best_emb = embedding
                    # Accept embedding only if IOU reasonable
                    if best_iou >= 0.3 and best_emb is not None:
                        embeddings[i] = best_emb
                        self.logger.debug(f"Direct embedding: Face {i} matched InsightFace embedding (IOU={best_iou:.2f})")
                    else:
                        self.logger.debug(f"Direct embedding: Face {i} had no matching InsightFace embedding (best IOU={best_iou:.2f})")
            except Exception as e:
                self.logger.debug(f"InsightFace mapping failed: {e}")
            
            # Detect humans
            humans = self.human_detector.detect_humans(processing_frame)
            
            # Update results
            with self.lock:
                self.current_faces = faces
                self.current_humans = humans
                self.current_embeddings = embeddings
                self.last_update_time = time.time()
            
            return {
                'faces': faces,
                'humans': humans,
                'embeddings': embeddings,
                'timestamp': time.time(),
                'frame_shape': frame.shape
            }
            
        except Exception as e:
            self.logger.error(f"Error in direct frame processing: {str(e)}")
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
                # Run InsightFace once on the full frame and map embeddings to detected boxes
                embeddings = [None] * len(faces)
                try:
                    proc_for_insight = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
                    insight_results = self.face_embedder.detect_and_encode_faces(proc_for_insight)

                    # Same IOU mapping as in direct processing
                    def to_xyxy(bbox):
                        x1, y1, x2, y2 = bbox
                        return [int(x1), int(y1), int(x2), int(y2)]

                    def iou(boxA, boxB):
                        ax, ay, aw, ah = boxA
                        a1x, a1y, a2x, a2y = ax, ay, ax+aw, ay+ah
                        bx1, by1, bx2, by2 = boxB
                        xA = max(a1x, bx1)
                        yA = max(a1y, by1)
                        xB = min(a2x, bx2)
                        yB = min(a2y, by2)
                        interW = max(0, xB - xA)
                        interH = max(0, yB - yA)
                        interArea = interW * interH
                        boxAArea = (a2x - a1x) * (a2y - a1y)
                        boxBArea = (bx2 - bx1) * (by2 - by1)
                        denom = float(boxAArea + boxBArea - interArea + 1e-6)
                        return interArea / denom if denom > 0 else 0.0

                    for i, face_box in enumerate(faces):
                        best_iou = 0.0
                        best_emb = None
                        bx, by, bw, bh, conf = face_box
                        for j, (face_info, embedding) in enumerate(insight_results):
                            if embedding is None:
                                continue
                            b_xyxy = to_xyxy(face_info['bbox'])
                            score = iou((bx, by, bw, bh), b_xyxy)
                            if score > best_iou:
                                best_iou = score
                                best_emb = embedding
                        if best_iou >= 0.3 and best_emb is not None:
                            embeddings[i] = best_emb
                            self.logger.debug(f"Face {i}: Matched InsightFace embedding (IOU={best_iou:.2f})")
                        else:
                            self.logger.debug(f"Face {i}: No matching InsightFace embedding (best IOU={best_iou:.2f})")
                except Exception as e:
                    self.logger.debug(f"InsightFace mapping failed in worker: {e}")
                
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
