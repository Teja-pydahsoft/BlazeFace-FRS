"""
Attendance Marking System for BlazeFace-FRS
Real-time attendance marking with face recognition
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
import logging

from ..core.dual_pipeline import DualPipeline
from ..core.database import DatabaseManager
from ..core.blazeface_detector import BlazeFaceDetector
from ..core.threshold_tuner import ThresholdTuner
from ..utils.camera_utils import CameraManager
from .face_tracker import FaceTracker
from ..core.insightface_embedder import InsightFaceEmbedder

class AttendanceMarkingDialog:
    def __init__(self, parent, database_manager: DatabaseManager, config: Dict[str, Any]):
        """
        Initialize attendance marking dialog
        
        Args:
            parent: Parent window
            database_manager: Database manager instance
            config: Application configuration
        """
        self.parent = parent
        self.database_manager = database_manager
        self.config = config
        
    # Initialize components
        camera_source = config.get('camera_sources', {}).get('webcam', config.get('camera_index', 0))
        self.camera_manager = CameraManager(camera_source)
        self.face_detector = BlazeFaceDetector(min_detection_confidence=0.7)  # Higher threshold for better accuracy
        self.pipeline = None
        self.face_embedder = None  # will be selected to match DB encodings
        
        # Initialize threshold tuner and face tracker
        self.threshold_tuner = ThresholdTuner(config, database_manager)
        self.face_tracker = FaceTracker(max_disappeared=10, max_distance=75)
        
        # Detect camera type for pipeline optimization
        self.camera_type = self._detect_camera_type(camera_source)
        
        # UI state
        self.is_running = False
        self.current_frame = None
        self.recognized_students = {}  # student_id -> (name, last_seen, confidence)
        self.attendance_marked = set()  # Set of student_ids already marked today
        self.student_names = {}  # Cache student names for quick lookup
        self.tracked_face_data = {}  # Stores recognition data for tracked faces
        # Consensus settings for auto-marking (N of M)
        # Example: require 3 consistent recognitions out of last 4 frames
        self.consensus_required = self.config.get('consensus_required', 3)
        self.consensus_window = self.config.get('consensus_window', 4)
        
        # Recognition consistency tracking
        self.last_recognition_result = None  # (student_id, confidence, timestamp)
        self.recognition_consistency_threshold = 0.05  # 5% variation allowed
        # Per-embedder recognition thresholds (can be overridden via config)
        # Example: use slightly higher threshold for 512-d InsightFace
        self.recognition_thresholds = config.get('recognition_thresholds', {
            'insightface': 0.62,
            'standard': 0.55,
            'simple': 0.5,
            'faiss': 0.60
        })
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Attendance Marking - BlazeFace-FRS")
        self.dialog.geometry("1000x700")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Setup UI
        self._setup_ui()
        
        # Load existing face encodings
        self._load_face_encodings()

        # Initialize embedder after loading encodings so we pick one matching stored types
        self._initialize_face_embedder()
        
        # Load today's attendance
        self._load_todays_attendance()
        
        # Start camera preview
        self._start_camera_preview()
        
        # Center dialog
        self._center_dialog()
    
    def _detect_camera_type(self, camera_source: str) -> str:
        """Detect camera type for pipeline optimization"""
        try:
            if isinstance(camera_source, int) or str(camera_source).isdigit():
                return "webcam"
            elif str(camera_source).startswith(("rtsp://", "http://", "https://")):
                return "stream"
            elif str(camera_source).endswith((".mp4", ".avi", ".mov", ".mkv")):
                return "video_file"
            else:
                return "webcam"
        except Exception as e:
            print(f"Error detecting camera type: {e}")
            return "webcam"
    
    def _setup_ui(self):
        """Setup the attendance marking UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls and Status
        self._setup_control_panel(main_frame)
        
        # Right panel - Camera and Recognition
        self._setup_camera_panel(main_frame)
        
        # Bottom panel - Attendance List
        self._setup_attendance_panel(main_frame)
    
    def _setup_control_panel(self, parent):
        """Setup control panel"""
        control_frame = ttk.LabelFrame(parent, text="Attendance Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(control_frame, text="Start Attendance", 
                                   command=self._start_attendance)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Attendance", 
                                  command=self._stop_attendance, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Recognition threshold (for face embedding similarity matching)
        ttk.Label(settings_frame, text="Recognition Threshold (face similarity):").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=self.config.get('recognition_confidence', 0.35))
        threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=0.8, 
                                  variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=(0, 5))
        
        # Camera selection
        ttk.Label(settings_frame, text="Camera Source:").pack(anchor=tk.W)
        self.camera_source_var = tk.StringVar(value="webcam")
        camera_frame = tk.Frame(settings_frame)
        camera_frame.pack(fill=tk.X, pady=(0, 5))
        
        camera_sources = self.config.get('camera_sources', {
            'webcam': 'Webcam (0)',
            'nvr_camera': 'NVR Camera (RTSP)',
            'ip_camera': 'IP Camera (HTTP)',
            'video_file': 'Video File'
        })
        
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_source_var, 
                                   values=list(camera_sources.keys()), state="readonly")
        camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(camera_frame, text="Switch", command=self._switch_camera).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Face detection confidence threshold (for filtering detected faces)
        ttk.Label(settings_frame, text="Min Face Detection Confidence (filter faces):").pack(anchor=tk.W)
        self.face_confidence_var = tk.DoubleVar(value=0.1)  # Very low threshold to detect more faces
        face_confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                        variable=self.face_confidence_var, orient=tk.HORIZONTAL)
        face_confidence_scale.pack(fill=tk.X, pady=(0, 5))
        
        # Auto-mark attendance
        self.auto_mark_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Auto-mark attendance", 
                       variable=self.auto_mark_var).pack(anchor=tk.W, pady=2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.total_students_label = ttk.Label(stats_frame, text="Total Students: 0")
        self.total_students_label.pack(anchor=tk.W)
        
        self.present_today_label = ttk.Label(stats_frame, text="Present Today: 0")
        self.present_today_label.pack(anchor=tk.W)
        
        self.recognized_now_label = ttk.Label(stats_frame, text="Recognized Now: 0")
        self.recognized_now_label.pack(anchor=tk.W)
        
        # Actions
        action_frame = ttk.LabelFrame(control_frame, text="Actions", padding="5")
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(action_frame, text="Refresh Students", 
                  command=self._refresh_students).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Test Recognition", 
                  command=self._test_recognition).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Export Attendance", 
                  command=self._export_attendance).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Clear Today's Attendance", 
                  command=self._clear_todays_attendance).pack(fill=tk.X, pady=2)
    
    def _setup_camera_panel(self, parent):
        """Setup camera display panel"""
        camera_frame = ttk.LabelFrame(parent, text="Live Recognition", padding="10")
        camera_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera display - Use Canvas for better image display
        self.camera_canvas = tk.Canvas(camera_frame, width=400, height=300, 
                                       background='black', highlightthickness=1,
                                       relief='sunken', bd=2)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add initial text overlay for status
        self.camera_canvas.create_text(200, 150, text="Initializing camera...", 
                                       fill='white', font=('Arial', 12), tags='status_text')
        
        # Force canvas to update and be visible
        self.camera_canvas.update_idletasks()
        
        # Ensure canvas is properly sized
        self._ensure_canvas_visible()
        
        # Recognition info
        self.recognition_info = tk.Text(camera_frame, height=8, width=50, wrap=tk.WORD)
        self.recognition_info.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = ttk.Label(camera_frame, text="Status: Ready", 
                                     font=('Arial', 9), foreground='blue')
        self.status_label.pack(fill=tk.X, pady=(5, 0))
    
    def _setup_attendance_panel(self, parent):
        """Setup attendance list panel"""
        attendance_frame = ttk.LabelFrame(parent, text="Today's Attendance", padding="10")
        attendance_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Create treeview for attendance list
        columns = ('Student ID', 'Name', 'Status', 'Time', 'Confidence')
        self.attendance_tree = ttk.Treeview(attendance_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        for col in columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=120, anchor=tk.CENTER)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(attendance_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load today's attendance
        self._load_todays_attendance()
    
    def _initialize_face_embedder(self):
        # Choose the embedder to match the database stored encodings to avoid dim mismatch.
        try:
            # Inspect DB encoding types
            rows = self.database_manager.get_face_encodings()
            type_counts = {}
            for _sid, _enc, enc_type, _img in rows:
                t = enc_type or 'insightface'
                type_counts[t] = type_counts.get(t, 0) + 1

            preferred = None
            if type_counts:
                # Pick the most common stored encoding type
                preferred = max(type_counts.items(), key=lambda x: x[1])[0]

            if preferred == 'standard':
                try:
                    from app.core.standard_face_embedder import StandardFaceEmbedder
                    self.face_embedder = StandardFaceEmbedder(model='large')
                    print("StandardFaceEmbedder selected to match DB (128-dim)")
                    return
                except Exception as e:
                    print(f"StandardFaceEmbedder init failed, will try InsightFace next: {e}")

            # Prefer InsightFace (512-d) as canonical runtime embedder
            try:
                self.face_embedder = InsightFaceEmbedder(model_name='buffalo_l')
                print("InsightFaceEmbedder (buffalo_l, 512-d) initialized for recognition and matching")
            except Exception as e:
                print(f"InsightFaceEmbedder failed: {e}")
                self.face_embedder = None
        except Exception as e:
            print(f"Error selecting embedder: {e}")
            try:
                self.face_embedder = InsightFaceEmbedder(model_name='buffalo_l')
            except Exception:
                self.face_embedder = None
    
    def _load_face_encodings(self):
        """Load face encodings from database"""
        try:
            self.face_encodings = self.database_manager.get_face_encodings()
            # student_id -> list of {'encoding': np.ndarray, 'encoding_type': str, 'image_path': str}
            self.student_encodings = {}
            self.student_names = {}  # student_id -> name
            
            for student_id, encoding, encoding_type, image_path in self.face_encodings:
                if student_id not in self.student_encodings:
                    self.student_encodings[student_id] = []
                    # Get student name and cache it
                    student_info = self.database_manager.get_student(student_id)
                    if student_info:
                        self.student_names[student_id] = student_info['name']
                    else:
                        self.student_names[student_id] = f"Student {student_id}"
                # Normalize and ensure dtype
                try:
                    if isinstance(encoding, np.ndarray):
                        enc_arr = encoding.astype(np.float32)
                    else:
                        enc_arr = np.asarray(encoding, dtype=np.float32)
                except Exception:
                    enc_arr = np.array(encoding, dtype=np.float32)

                self.student_encodings[student_id].append({
                    'encoding': enc_arr,
                    'encoding_type': encoding_type or 'insightface',
                    'image_path': image_path
                })
            
            print(f"Loaded {len(self.face_encodings)} face encodings for {len(self.student_encodings)} students")
            
            # Debug: Show loaded encodings
            for student_id, enc_list in self.student_encodings.items():
                print(f"  Student {student_id}: {len(enc_list)} encodings")
                for i, enc_obj in enumerate(enc_list):
                    encoding = enc_obj['encoding']
                    enc_type = enc_obj.get('encoding_type', 'insightface')
                    print(f"    Encoding {i} (type={enc_type}): shape={encoding.shape}, norm={np.linalg.norm(encoding):.4f}")
                    print(f"    Encoding {i}: sample values={encoding[:5]}")
                
                # Check if encodings are different
                if len(enc_list) > 1:
                    from app.core.simple_face_embedder import SimpleFaceEmbedder
                    embedder = SimpleFaceEmbedder()
                    is_same, similarity = embedder.compare_faces(enc_list[0]['encoding'], enc_list[1]['encoding'], 0.9)
                    print(f"    Student {student_id}: Encoding 0 vs 1: is_same={is_same}, similarity={similarity:.4f}")
            
        except Exception as e:
            print(f"Error loading face encodings: {e}")
            self.face_encodings = []
            self.student_encodings = {}
            self.student_names = {}
    
    def _load_todays_attendance(self):
        """Load today's attendance records"""
        try:
            from datetime import datetime
            today = datetime.now().date().strftime('%Y-%m-%d')
            
            records = self.database_manager.get_attendance_records(date_from=today, date_to=today)
            
            # Clear existing items
            for item in self.attendance_tree.get_children():
                self.attendance_tree.delete(item)
            
            # Add records to treeview
            for record in records:
                status = "Present" if record['status'] == 'present' else record['status'].title()
                # Fix the time formatting issue - ensure time is properly converted to string
                time_str = str(record['time']) if record['time'] else "N/A"
                
                # Handle confidence field - it might be stored as bytes or float
                confidence = "N/A"
                if record['confidence'] is not None:
                    try:
                        if isinstance(record['confidence'], bytes):
                            # Convert bytes to float
                            import struct
                            confidence_val = struct.unpack('f', record['confidence'])[0]
                            confidence = f"{confidence_val:.2f}"
                        else:
                            # Already a float
                            confidence = f"{record['confidence']:.2f}"
                    except Exception as e:
                        print(f"Error processing confidence value: {e}")
                        confidence = "N/A"
                
                self.attendance_tree.insert('', 'end', values=(
                    record['student_id'],
                    record['name'],
                    status,
                    time_str,
                    confidence
                ))
            
            # Update statistics
            self._update_statistics()
            
        except Exception as e:
            print(f"Error loading today's attendance: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_camera_preview(self):
        """Start camera preview"""
        try:
            if self.camera_manager.is_camera_available():
                self._update_camera_preview()
            else:
                self.camera_canvas.delete('all')
                self.camera_canvas.create_text(250, 200, text="Camera not available", 
                                              fill='red', font=('Arial', 12), tags='status_text')
                self.status_label.config(text="Status: Camera not available", foreground='red')
        except Exception as e:
            print(f"Error starting camera preview: {e}")
            self.camera_canvas.delete('all')
            self.camera_canvas.create_text(250, 200, text=f"Camera error: {str(e)}", 
                                          fill='red', font=('Arial', 12), tags='status_text')
            self.status_label.config(text=f"Status: Error - {str(e)}", foreground='red')
    
    def _ensure_canvas_visible(self):
        """Ensure canvas is properly sized and visible"""
        try:
            # Force canvas to be visible and properly sized
            self.camera_canvas.update_idletasks()
            
            # Get actual canvas dimensions
            canvas_width = self.camera_canvas.winfo_width()
            canvas_height = self.camera_canvas.winfo_height()
            
            # If canvas is too small, resize it
            if canvas_width < 100 or canvas_height < 100:
                self.camera_canvas.config(width=500, height=400)
                self.camera_canvas.update_idletasks()
                # Canvas resized for attendance marking
            
            # Create a test rectangle to verify canvas is working
            self.camera_canvas.delete('all')
            self.camera_canvas.create_rectangle(10, 10, canvas_width-10, canvas_height-10, 
                                              outline='white', width=2, fill='black')
            self.camera_canvas.create_text(canvas_width//2, canvas_height//2, 
                                          text="Camera Initializing...", 
                                          fill='white', font=('Arial', 12))
            self.camera_canvas.update_idletasks()
            
        except Exception as e:
            print(f"Error ensuring canvas visibility: {e}")
    
    def _update_camera_preview(self):
        """Update camera preview with performance optimizations"""
        try:
            if self.camera_manager.is_camera_available():
                ret, frame = self.camera_manager.get_frame()
                if ret and frame is not None:
                    # Performance optimization: Skip frames for recognition processing
                    if not hasattr(self, '_frame_skip_counter'):
                        self._frame_skip_counter = 0
                    if not hasattr(self, '_last_recognition_time'):
                        self._last_recognition_time = 0
                    
                    self._frame_skip_counter += 1
                    current_time = time.time()
                    
                    # Process frame through pipeline if running
                    if self.is_running and self.pipeline:
                        # Initialize results with empty data
                        results = {'faces': [], 'embeddings': []}
                        
                        # Only process every few frames for better performance
                        if self._frame_skip_counter % 3 == 0:  # Process every 3rd frame
                            results = self.pipeline.process_frame(frame)
                            
                            # Debug: Log detection results
                            faces_detected = len(results.get('faces', []))
                            if faces_detected > 0:
                                print(f"Detection: Found {faces_detected} faces")
                                for i, face in enumerate(results.get('faces', [])):
                                    x, y, w, h, conf = face
                                    print(f"  Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                            else:
                                # No faces detected - removed verbose logging
                                pass
                            
                            frame = self._draw_enhanced_detections(frame, results)
                        
                        # Only do recognition processing at intervals
                        if current_time - self._last_recognition_time > 0.5:  # 0.5 second intervals
                            self._process_recognition_results(results)
                            self._last_recognition_time = current_time
                        else:
                            # Just draw existing detections without processing
                            frame = self._draw_enhanced_detections(frame, results)
                    elif self.is_running and self.camera_type == 'stream':
                        # Fallback: Direct face detection for NVR cameras if pipeline fails
                        if self._frame_skip_counter % 5 == 0:  # Even less frequent for fallback
                            faces = self.face_detector.detect_faces(frame)
                            if faces:
                                # Direct NVR detection completed - removed verbose logging
                                pass
                            
                            # Draw faces directly
                            frame = self.face_detector.draw_faces(frame, faces)
                    
                    # Resize frame for display
                    display_frame = cv2.resize(frame, (500, 400))
                    
                    # Convert frame for display using Canvas
                    try:
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        
                        # Clear canvas and display image
                        self.camera_canvas.delete('all')
                        self.camera_canvas.create_image(250, 200, image=frame_tk, anchor=tk.CENTER)
                        self.camera_canvas.image = frame_tk  # Keep reference to prevent garbage collection
                        
                        # Force canvas update
                        self.camera_canvas.update_idletasks()
                        
                    except Exception as img_error:
                        # Fallback: show text on canvas
                        self.camera_canvas.delete('all')
                        self.camera_canvas.create_text(250, 200, text=f"Camera Active\nFrame: {frame.shape}", 
                                                      fill='green', font=('Arial', 12), tags='status_text')
                    
                    self.current_frame = frame
                else:
                    # Only show "No camera feed" if we've been trying for a while
                    if not hasattr(self, '_no_feed_count'):
                        self._no_feed_count = 0
                    self._no_feed_count += 1
                    
                    if self._no_feed_count > 15:  # Increased threshold for more stability
                        self.camera_canvas.delete('all')
                        self.camera_canvas.create_text(250, 200, text="No camera feed", 
                                                      fill='red', font=('Arial', 12), tags='status_text')
                        self.status_label.config(text="Status: No camera feed", foreground='red')
            else:
                # Camera not available, try to reinitialize much less frequently
                if not hasattr(self, '_reinit_count'):
                    self._reinit_count = 0
                if not hasattr(self, '_last_reinit_time'):
                    self._last_reinit_time = 0
                
                current_time = time.time()
                self._reinit_count += 1
                
                # Only try to reinitialize every 60 seconds and after 30 failed attempts
                if (self._reinit_count > 30 and 
                    current_time - self._last_reinit_time > 60):
                    print("Camera not available, attempting to reinitialize...")
                    try:
                        self.camera_manager.release()
                        camera_source = self.config.get('camera_sources', {}).get('webcam', self.config.get('camera_index', 0))
                        from ..utils.camera_utils import CameraManager
                        self.camera_manager = CameraManager(camera_source)
                        self._reinit_count = 0
                        self._last_reinit_time = current_time
                    except Exception as e:
                        print(f"Failed to reinitialize camera: {e}")
                        self._last_reinit_time = current_time
            
            # Schedule next update with adaptive timing
            update_interval = 30 if self.is_running else 50  # Faster when running, slower when idle
            self.dialog.after(update_interval, self._update_camera_preview)
            
        except Exception as e:
            print(f"Error updating camera preview: {e}")
            self.camera_canvas.delete('all')
            self.camera_canvas.create_text(250, 200, text=f"Camera error: {str(e)}", 
                                          fill='red', font=('Arial', 12), tags='status_text')
            self.status_label.config(text=f"Status: Error - {str(e)}", foreground='red')
            self.dialog.after(1000, self._update_camera_preview)
    
    def _draw_enhanced_detections(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Draw enhanced detections using the face tracker's output."""
        try:
            result_frame = frame.copy()
            
            # Use the tracker's current state to draw
            for (object_id, centroid) in self.face_tracker.faces.items():
                # Find the original bounding box for this centroid
                best_rect = None
                min_dist = float('inf')
                for r in results.get('faces', []):
                    x,y,w,h,conf = r
                    cx, cy = x + w // 2, y + h // 2
                    dist = np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_rect = (x,y,w,h)

                if best_rect is None:
                    continue

                x, y, w, h = best_rect
                
                # Get recognition data from our tracked data store
                track_data = self.tracked_face_data.get(object_id)
                
                student_name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                
                if track_data and track_data['student_id']:
                    student_id = track_data['student_id']
                    confidence = track_data['confidence']
                    student_name = f"{self.student_names.get(student_id, student_id)} ({confidence:.2f})"
                    color = (0, 255, 0) # Green for recognized
                
                # Draw the bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw the tracker ID and the student name
                text = f"ID {object_id}: {student_name}"
                cv2.putText(result_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            return result_frame

        except Exception as e:
            print(f"Error drawing enhanced detections: {e}")
            return frame
    
    def _process_recognition_results(self, results: Dict[str, Any]):
        """Process recognition results using the face tracker and mark attendance."""
        try:
            faces = results.get('faces', [])
            embeddings = results.get('embeddings', [])
            
            rects = [tuple(face[:4]) for face in faces]
            
            tracked_objects = self.face_tracker.update(rects)

            for (object_id, centroid) in tracked_objects.items():
                # Ensure tracked_face_data has a recognition history buffer
                if object_id not in self.tracked_face_data:
                    self.tracked_face_data[object_id] = {
                        'student_id': None,
                        'confidence': 0.0,
                        'last_seen': time.time(),
                        'history': []  # list of (student_id, confidence, timestamp)
                    }

                if self.tracked_face_data[object_id]['student_id'] is None:
                    
                    best_face_idx = -1
                    min_dist = float('inf')
                    for i, rect in enumerate(rects):
                        x, y, w, h = rect
                        face_centroid_x, face_centroid_y = x + w // 2, y + h // 2
                        dist = np.sqrt((face_centroid_x - centroid[0])**2 + (face_centroid_y - centroid[1])**2)
                        if dist < min_dist and dist < 50:
                            min_dist = dist
                            best_face_idx = i

                    if best_face_idx != -1 and best_face_idx < len(embeddings):
                        embedding = embeddings[best_face_idx]
                        if embedding is not None:
                            print(f"Tracker ID {object_id}: New face detected. Running recognition.")
                            best_match = self._find_best_match(embedding)
                            
                            student_id = None
                            recognition_confidence = 0.0
                            
                            if best_match:
                                student_id, recognition_confidence = best_match
                                print(f"Tracker ID {object_id}: Matched student {student_id} with confidence {recognition_confidence:.4f}")
                                # Append to recognition history for consensus
                                hist = self.tracked_face_data[object_id].get('history', [])
                                hist.append((student_id, recognition_confidence, time.time()))
                                # Trim history to window
                                hist = hist[-self.consensus_window:]
                                self.tracked_face_data[object_id]['history'] = hist

                                # Try consensus-based auto-mark
                                if self.auto_mark_var.get():
                                    self._try_consensus_mark(object_id)
                            else:
                                print(f"Tracker ID {object_id}: No match found.")
                            # Update last_seen and keep stored student_id/confidence for display
                            self.tracked_face_data[object_id]['last_seen'] = time.time()
                            self.tracked_face_data[object_id]['student_id'] = student_id
                            self.tracked_face_data[object_id]['confidence'] = recognition_confidence
                        else:
                            self.tracked_face_data[object_id]['last_seen'] = time.time()
                            self.tracked_face_data[object_id]['student_id'] = None
                            self.tracked_face_data[object_id]['confidence'] = 0.0
                    else:
                        self.tracked_face_data[object_id]['last_seen'] = time.time()
                        self.tracked_face_data[object_id]['student_id'] = None
                        self.tracked_face_data[object_id]['confidence'] = 0.0
                else:
                    self.tracked_face_data[object_id]['last_seen'] = time.time()

            current_tracked_ids = set(tracked_objects.keys())
            all_known_ids = set(self.tracked_face_data.keys())
            disappeared_ids = all_known_ids - current_tracked_ids
            for object_id in disappeared_ids:
                if time.time() - self.tracked_face_data[object_id]['last_seen'] > 2:
                    print(f"Tracker ID {object_id}: Removing stale track.")
                    del self.tracked_face_data[object_id]

            self._update_recognition_display()

        except Exception as e:
            print(f"Error processing recognition results: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_best_match(self, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Find best matching student for the given embedding with strict quality checks"""
        try:
            if self.face_embedder is None:
                print("Face embedder not initialized")
                return None
            # Fast-path: if a FAISS index is loaded in the pipeline and the query dim matches, use it
            all_similarities = []  # list of (student_id, enc_idx, sim, encoding_type)
            best_similarity = 0.0
            best_student_id = None
            best_encoding_type = None

            faiss_index = None
            try:
                if hasattr(self, 'pipeline') and getattr(self.pipeline, 'faiss_index', None) is not None:
                    faiss_index = self.pipeline.faiss_index
            except Exception:
                faiss_index = None

            if faiss_index is not None and faiss_index.dim is not None:
                try:
                    q = np.asarray(query_embedding, dtype=np.float32)
                    if q.ndim == 1:
                        q = q.reshape(1, -1)
                    if q.shape[1] == faiss_index.dim:
                        # Search top-k from FAISS
                        k = min(5, len(faiss_index.names) if faiss_index.names is not None else 5)
                        D, I, N = faiss_index.search(q, k=k)
                        # D: similarities, N: names list
                        sims = D[0].tolist()
                        names = N[0]
                        # Aggregate per-student max similarity
                        per_student = {}
                        for idx, (sim, name) in enumerate(zip(sims, names)):
                            if name is None:
                                continue
                            sid = str(name)
                            # We don't have enc_idx from FAISS; use -1 as placeholder
                            all_similarities.append((sid, -1, float(sim), 'faiss'))
                            if sid not in per_student or per_student[sid] < sim:
                                per_student[sid] = float(sim)

                        # Find best and second best
                        sorted_per = sorted(per_student.items(), key=lambda x: x[1], reverse=True)
                        if sorted_per:
                            best_student_id, best_similarity = sorted_per[0]
                            best_encoding_type = 'faiss'
                        # For debug, print top few
                        print(f"FAISS fast-path returned top {len(sorted_per)} candidates")
                    else:
                        print(f"FAISS skip: dimension mismatch (index={faiss_index.dim}, query={q.shape[1]})")
                        faiss_index = None
                except Exception as e:
                    print(f"FAISS search failed: {e}")
                    faiss_index = None

            # If FAISS didn't produce results, fall back to the brute-force matching below
            if faiss_index is None:
                print(f"Comparing against {len(self.student_encodings)} students (brute-force)")
                for student_id, enc_list in self.student_encodings.items():
                    print(f"Checking student {student_id} with {len(enc_list)} encodings")
                    student_max_similarity = 0.0

                    for i, enc_obj in enumerate(enc_list):
                        encoding = enc_obj.get('encoding')
                        enc_type = enc_obj.get('encoding_type', 'insightface')

                        # Skip if encoding dims don't match query
                        try:
                            if encoding is None or encoding.ndim != 1 or encoding.shape[0] != query_embedding.shape[0]:
                                print(f"  Skipping encoding {i} for student {student_id}: shape mismatch (stored={None if encoding is None else encoding.shape}, query={query_embedding.shape})")
                                continue
                        except Exception:
                            continue

                        # Use compare_faces to get similarity (we pass very low threshold to obtain the raw score)
                        is_same, similarity = self.face_embedder.compare_faces(query_embedding, encoding, 0.01)
                        all_similarities.append((student_id, i, similarity, enc_type))
                        # Track the highest similarity for this student
                        if similarity > student_max_similarity:
                            student_max_similarity = similarity
                        print(f"  Encoding {i} (type={enc_type}): similarity={similarity:.4f}")

                    print(f"  Student {student_id} max similarity: {student_max_similarity:.4f}")

                    # Track the best match regardless of threshold (threshold check happens later)
                    if student_max_similarity > best_similarity:
                        best_similarity = student_max_similarity
                        best_student_id = student_id
                        # find which encoding produced the student_max_similarity to know its encoding_type
                        # search all_similarities for that student
                        best_enc_type = None
                        for sid, enc_idx, sim, enc_type in all_similarities:
                            if sid == student_id and abs(sim - student_max_similarity) < 1e-6:
                                best_enc_type = enc_type
                                break
                        best_encoding_type = best_enc_type or 'insightface'
                        print(f"  -> New best match: {student_id} with similarity {student_max_similarity:.4f} (encoding_type={best_encoding_type})")
            
            # Debug: Show all similarities
            print("All similarities:")
            for entry in sorted(all_similarities, key=lambda x: x[2], reverse=True)[:5]:
                student_id, enc_idx, sim, enc_type = entry
                print(f"  {student_id}[{enc_idx}] ({enc_type}): {sim:.4f}")
            
            print(f"Final result: best_similarity={best_similarity:.4f}, best_student_id={best_student_id}")
            
            # Calculate gap between best and second best (needed for all logic paths)
            second_best = 0.0
            for entry in sorted(all_similarities, key=lambda x: x[2], reverse=True)[1:3]:
                _, _, sim, _ = entry
                if sim > second_best:
                    second_best = sim
            
            gap = best_similarity - second_best
            min_gap = self.config.get('min_confidence_gap', 0.05)  # Increased to 0.05 for better discrimination
            
            # SPECIAL CHECK: Handle known problematic pairs
            if best_student_id in ['1233', '52856'] and best_similarity > 0.90:
                # Check if the second best is the other problematic student
                second_best_student = None
                second_best_similarity = 0.0
                
                for student_id, enc_idx, sim, _ in sorted(all_similarities, key=lambda x: x[2], reverse=True)[1:3]:
                    if sim > second_best_similarity and student_id != best_student_id:
                        second_best_similarity = sim
                        second_best_student = student_id
                
                # If second best is the other problematic student and gap is small, reject
                if (second_best_student in ['1233', '52856'] and 
                    best_similarity - second_best_similarity < 0.10):
                    print(f"⚠️  PROBLEMATIC PAIR DETECTED: {best_student_id} vs {second_best_student}")
                    print(f"⚠️  Similarity gap too small: {best_similarity:.4f} - {second_best_similarity:.4f} = {best_similarity - second_best_similarity:.4f}")
                    print(f"⚠️  REJECTING MATCH - This face will be marked as UNKNOWN")
                    return None
            
            # Determine recognition threshold based on encoding type of best match
            if best_encoding_type:
                recognition_threshold = self.recognition_thresholds.get(best_encoding_type, self.config.get('recognition_confidence', 0.6))
            else:
                recognition_threshold = self.config.get('recognition_confidence', 0.6)

            print(f"Using recognition threshold for encoding '{best_encoding_type}': {recognition_threshold:.2f}")

            # QUALITY CHECK: Accept matches above threshold, or best match if no one meets threshold (fallback handled later)
            if best_similarity >= recognition_threshold:
                
                # SMART GAP LOGIC: Adjust gap requirement based on number of students
                unique_students = len(set(sid for sid, _, _ in all_similarities))
                
                if unique_students == 1:
                    # Single student scenario: Only require threshold, ignore gap
                    gap_required = 0.0
                    print(f"ℹ️  SINGLE STUDENT SCENARIO: Ignoring gap requirement")
                else:
                    # Multiple students: Use normal gap requirement
                    gap_required = min_gap
                
                if best_similarity >= recognition_threshold and gap >= gap_required:
                    # Additional consistency check: Compare with last recognition result
                    current_time = time.time()
                    if self.last_recognition_result:
                        last_student_id, last_confidence, last_time = self.last_recognition_result
                        time_diff = current_time - last_time
                        
                        # If recognition happened recently (within 1 second) and results are inconsistent
                        if time_diff < 1.0:  # 1.0 seconds
                            if last_student_id != best_student_id:
                                # Only reject if confidence difference is very large
                                confidence_diff = abs(best_similarity - last_confidence)
                                if confidence_diff > 0.1:
                                    print(f"\u26a0\ufe0f  INCONSISTENT RECOGNITION: Last result was {last_student_id} ({last_confidence:.4f}), now {best_student_id} ({best_similarity:.4f})")
                                    print(f"\u26a0\ufe0f  REJECTING INCONSISTENT MATCH - This face will be marked as UNKNOWN")
                                    return None
                                else:
                                    print(f"\u2139\ufe0f  SIMILAR STUDENTS: Switching from {last_student_id} to {best_student_id} (diff: {confidence_diff:.4f})")
                            elif abs(best_similarity - last_confidence) > self.recognition_consistency_threshold:
                                print(f"\u26a0\ufe0f  CONFIDENCE VARIATION: Last {last_confidence:.4f}, now {best_similarity:.4f} (diff: {abs(best_similarity - last_confidence):.4f})")
                                print(f"\u26a0\ufe0f  REJECTING VARIABLE MATCH - This face will be marked as UNKNOWN")
                                return None
                    
                    # Store this recognition result for consistency checking
                    self.last_recognition_result = (best_student_id, best_similarity, current_time)
                    
                    print(f"\u2713 CLEAR MATCH FOUND: {best_student_id} with similarity {best_similarity:.4f} (gap: {gap:.4f})")
                    return best_student_id, best_similarity
                else:
                    print(f"\u2717 INSUFFICIENT CONFIDENCE: {best_similarity:.4f} < {recognition_threshold:.2f} OR gap {gap:.4f} < {gap_required:.3f}")
                    # Tightened fallback: accept only if best_similarity >= fallback_min and gap >= fallback_gap and
                    # the best_encoding_type permits fallback (configurable). Default: no fallback.
                    fallback_min = self.config.get('fallback_min_similarity', None)
                    fallback_allowed_types = self.config.get('fallback_allowed_encoding_types', [])

                    if fallback_min is not None and best_similarity >= fallback_min and gap >= (self.config.get('fallback_min_gap', 0.03)):
                        if not fallback_allowed_types or (best_encoding_type in fallback_allowed_types):
                            print(f"\u26a0\ufe0f  FALLBACK MATCH ACCEPTED (type={best_encoding_type}) with similarity {best_similarity:.4f}")
                            # store recognition
                            self.last_recognition_result = (best_student_id, best_similarity, time.time())
                            return best_student_id, best_similarity
                    print(f"\u2717 REJECTING MATCH - This face will be marked as UNKNOWN")
                    return None
            else:
                # Fallback: If best match is below threshold but still reasonable, accept it with lower confidence
                # Adjust gap requirement based on number of students
                unique_students = len(set(sid for sid, _, _ in all_similarities))
                fallback_gap_required = 0.03 if unique_students > 1 else 0.0
                
                if best_similarity >= 0.35 and gap >= fallback_gap_required:  # Smart gap requirement
                    print(f"⚠️  FALLBACK MATCH: Best similarity {best_similarity:.4f} below threshold {recognition_threshold:.2f} but above 0.35 with gap {gap:.3f}")
                    print(f"⚠️  ACCEPTING FALLBACK MATCH: {best_student_id} with similarity {best_similarity:.4f}")
                    return best_student_id, best_similarity
                else:
                    print(f"✗ NO MATCH: Best similarity {best_similarity:.4f} below recognition threshold {recognition_threshold:.2f}")
                    print(f"✗ NO MATCH: Gap {gap:.3f} insufficient for reliable discrimination (required: {fallback_gap_required:.3f})")
                    print(f"✗ NO MATCH: This face will be marked as UNKNOWN")
                    return None
            
        except Exception as e:
            print(f"Error finding best match: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _switch_camera(self):
        """Switch to selected camera source"""
        try:
            selected_source = self.camera_source_var.get()
            camera_sources = self.config.get('camera_sources', {})
            
            if selected_source not in camera_sources:
                messagebox.showerror("Error", f"Camera source '{selected_source}' not found in configuration")
                return
            
            camera_source = camera_sources[selected_source]
            
            # Stop current camera if running
            if self.is_running:
                self._stop_attendance()
            
            # Release current camera
            if self.camera_manager:
                self.camera_manager.release()
            
            # Initialize new camera
            from ..utils.camera_utils import CameraManager
            self.camera_manager = CameraManager(camera_source)
            
            # Update camera type
            self.camera_type = self._detect_camera_type(camera_source)
            
            if self.camera_manager.is_initialized:
                messagebox.showinfo("Success", f"Switched to {selected_source} camera ({self.camera_type})")
                print(f"Switched to camera: {camera_source} (type: {self.camera_type})")
            else:
                messagebox.showerror("Error", f"Failed to initialize {selected_source} camera")
                print(f"Failed to initialize camera: {camera_source}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to switch camera: {str(e)}")
            print(f"Error switching camera: {e}")

    def _mark_attendance(self, student_id: str, recognition_confidence: float):
        """Mark attendance for a student"""
        try:
            from datetime import datetime
            
            # Check if already marked today
            today = datetime.now().date().strftime('%Y-%m-%d')
            if student_id in self.attendance_marked:
                print(f"Student {student_id} already marked today, skipping")
                return
            
            print(f"Attempting to mark attendance for {student_id} with recognition confidence {recognition_confidence:.4f}")
            
            # Mark attendance with face recognition confidence (similarity score)
            success = self.database_manager.add_attendance_record(
                student_id=student_id,
                status='present',
                confidence=recognition_confidence,  # This is the face embedding similarity score
                detection_type='face_recognition',
                notes=f"Auto-marked via face recognition with similarity {recognition_confidence:.4f}"
            )
            
            if success:
                self.attendance_marked.add(student_id)
                print(f"✓ Successfully marked attendance for {student_id}")
                self._log_message(f"✓ Attendance marked for {student_id}")
                self._load_todays_attendance()  # Refresh the list
            else:
                print(f"✗ Failed to mark attendance for {student_id}")
                self._log_message(f"✗ Failed to mark attendance for {student_id}")
                
        except Exception as e:
            print(f"Error marking attendance: {e}")
            import traceback
            traceback.print_exc()
            self._log_message(f"✗ Error marking attendance: {str(e)}")
    
    def _try_consensus_mark(self, object_id: int):
        """Try to mark attendance for a tracked object if N-of-M consensus is met."""
        try:
            data = self.tracked_face_data.get(object_id)
            if not data:
                return

            history = data.get('history', [])
            if not history:
                return

            # Count votes for top student_id in history window
            counts = {}
            for sid, conf, ts in history:
                if sid is None:
                    continue
                counts[sid] = counts.get(sid, 0) + 1

            if not counts:
                return

            # Find best voted student and its count
            best_student_id, votes = max(counts.items(), key=lambda x: x[1])
            if votes >= self.consensus_required:
                # Find latest confidence for that student in history
                latest_conf = 0.0
                for sid, conf, ts in reversed(history):
                    if sid == best_student_id:
                        latest_conf = conf
                        break

                # Confirm not already marked today
                if best_student_id in self.attendance_marked:
                    print(f"Student {best_student_id} already marked today (consensus).")
                    return

                print(f"Consensus reached for tracker {object_id}: {best_student_id} with {votes}/{len(history)} votes")
                self._mark_attendance(best_student_id, latest_conf)

        except Exception as e:
            print(f"Error in consensus marking: {e}")
            import traceback
            traceback.print_exc()

    def _update_recognition_display(self):
        """Update the recognition information display"""
        try:
            info_text = f"Recognition Status:\n"
            info_text += f"Active Students: {len(self.recognized_students)}\n"
            info_text += f"Faces Detected: {len(self.recognized_students)}\n\n"
            
            if self.recognized_students:
                info_text += "Recognized Students:\n"
                for student_id, (name, last_seen, confidence) in self.recognized_students.items():
                    time_ago = time.time() - last_seen
                    if time_ago < 5:  # Only show recently recognized
                        info_text += f"• {name} ({student_id}) - {confidence:.2f}\n"
            else:
                info_text += "No students recognized yet\n"
            
            self.recognition_info.delete(1.0, tk.END)
            self.recognition_info.insert(1.0, info_text)
            
        except Exception as e:
            print(f"Error updating recognition display: {e}")
    
    def _update_statistics(self):
        """Update statistics display"""
        try:
            # Get total students
            all_students = self.database_manager.get_all_students()
            total_students = len(all_students)
            
            # Get present today from database
            from datetime import datetime
            today = datetime.now().date().strftime('%Y-%m-%d')
            today_records = self.database_manager.get_attendance_records(date_from=today, date_to=today)
            unique_present = len(set(record['student_id'] for record in today_records))
            
            # Get currently recognized
            recognized_now = len([s for s in self.recognized_students.values() 
                                if time.time() - s[1] < 5])
            
            self.total_students_label.config(text=f"Total Students: {total_students}")
            self.present_today_label.config(text=f"Present Today: {unique_present}")
            self.recognized_now_label.config(text=f"Marked This Session: {len(self.attendance_marked)}")
            
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def _log_message(self, message: str):
        """Log message to recognition info"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            self.recognition_info.insert(tk.END, log_message)
            self.recognition_info.see(tk.END)
        except Exception as e:
            print(f"Error logging message: {e}")
    
    def _start_attendance(self):
        """Start attendance marking"""
        try:
            if self.is_running:
                return
            
            # Initialize pipeline with camera type and correct embedder
            pipeline_config = self.config.copy()
            pipeline_config['camera_type'] = self.camera_type
            self.pipeline = DualPipeline(pipeline_config, face_embedder=self.face_embedder)
            self.pipeline.start_pipeline()
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.status_label.config(text="Status: Attendance marking active - Auto-marking enabled", foreground='green')
            self._log_message("Attendance marking started - Auto-marking enabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start attendance marking: {str(e)}")
    
    def _stop_attendance(self):
        """Stop attendance marking"""
        try:
            if not self.is_running:
                return
            
            if self.pipeline:
                self.pipeline.stop_pipeline()
                self.pipeline = None
            
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            self.status_label.config(text="Status: Attendance marking stopped", foreground='blue')
            self._log_message("Attendance marking stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop attendance marking: {str(e)}")
    
    def _refresh_students(self):
        """Refresh student data and face encodings"""
        try:
            self._load_face_encodings()
            self._load_todays_attendance()
            self._log_message("Student data refreshed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh students: {str(e)}")
    
    def _test_recognition(self):
        """Test recognition with current frame"""
        try:
            if self.current_frame is None:
                messagebox.showwarning("Warning", "No camera feed available")
                return
            
            if not self.student_encodings:
                messagebox.showwarning("Warning", "No student encodings available")
                return
            
            # Detect faces in current frame
            faces = self.face_detector.detect_faces(self.current_frame)
            if not faces:
                messagebox.showwarning("Warning", "No face detected in current frame")
                return
            
            # Use first detected face
            face_box = faces[0]
            x, y, w, h, confidence = face_box
            face_region = self.face_detector.extract_face_region(self.current_frame, (x, y, w, h))
            
            if face_region is None:
                messagebox.showerror("Error", "Failed to extract face region")
                return
            
            # Get embedding
            embedding = self.face_embedder.get_embedding(face_region)
            if embedding is None:
                messagebox.showerror("Error", "Failed to get face embedding")
                return
            
            # Test with different thresholds
            results = []
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            for threshold in thresholds:
                best_confidence = 0.0
                best_student_id = None
                
                for student_id, encodings in self.student_encodings.items():
                    for stored_obj in encodings:
                        stored_encoding = stored_obj.get('encoding')
                        # Skip mismatched dims
                        if stored_encoding is None or stored_encoding.shape[0] != embedding.shape[0]:
                            continue
                        is_same, similarity = self.face_embedder.compare_faces(embedding, stored_encoding, threshold)
                        if is_same and similarity > best_confidence:
                            best_confidence = similarity
                            best_student_id = student_id
                
                if best_student_id:
                    results.append(f"Threshold {threshold:.1f}: MATCH {best_student_id} (confidence: {best_confidence:.4f})")
                else:
                    results.append(f"Threshold {threshold:.1f}: No match")
            
            # Show results
            result_text = "Recognition Test Results:\n\n" + "\n".join(results)
            messagebox.showinfo("Recognition Test", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition test failed: {str(e)}")
            print(f"Recognition test error: {e}")
            import traceback
            traceback.print_exc()
    
    def _export_attendance(self):
        """Export today's attendance to CSV"""
        try:
            from datetime import datetime
            import csv
            
            today = datetime.now().date().strftime('%Y-%m-%d')
            records = self.database_manager.get_attendance_records(date_from=today, date_to=today)
            
            if not records:
                messagebox.showinfo("Info", "No attendance records found for today")
                return
            
            # Create filename
            filename = f"attendance_{today}.csv"
            
            # Write CSV file
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Student ID', 'Name', 'Department', 'Date', 'Time', 'Status', 'Confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in records:
                    writer.writerow({
                        'Student ID': record['student_id'],
                        'Name': record['name'],
                        'Department': record.get('department', ''),
                        'Date': record['date'],
                        'Time': record['time'],
                        'Status': record['status'],
                        'Confidence': record.get('confidence', '')
                    })
            
            messagebox.showinfo("Success", f"Attendance exported to {filename}")
            self._log_message(f"Attendance exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export attendance: {str(e)}")
    
    def _clear_todays_attendance(self):
        """Clear today's attendance records"""
        try:
            if messagebox.askyesno("Confirm", "Are you sure you want to clear today's attendance records?"):
                from datetime import datetime
                today = datetime.now().date().strftime('%Y-%m-%d')
                
                # Delete today's attendance records
                success = self.database_manager.delete_attendance_records(
                    date_from=today, 
                    date_to=today
                )
                
                if success:
                    self.attendance_marked.clear()
                    self._load_todays_attendance()
                    messagebox.showinfo("Success", "Today's attendance records cleared")
                    self._log_message("Today's attendance records cleared")
                    print("✓ Today's attendance cleared for testing")
                else:
                    messagebox.showwarning("Warning", "No attendance records found for today")
                    print("✗ No attendance records found for today")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear attendance: {str(e)}")
    
    def _center_dialog(self):
        """Center the dialog on the parent window"""
        try:
            self.dialog.update_idletasks()
            width = self.dialog.winfo_width()
            height = self.dialog.winfo_height()
            x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
            self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        except:
            pass
    
    def _cancel(self):
        """Cancel and close dialog"""
        try:
            if self.is_running:
                self._stop_attendance()
            self.camera_manager.release()
            self.dialog.destroy()
        except Exception as e:
            print(f"Error closing dialog: {e}")
    
    def __del__(self):
        """Destructor"""
        try:
            if hasattr(self, 'camera_manager'):
                self.camera_manager.release()
            if hasattr(self, 'face_detector'):
                self.face_detector.release()
            if hasattr(self, 'face_embedder') and self.face_embedder:
                self.face_embedder.release()
        except:
            pass
    
    def _process_frame_for_recognition(self, frame: np.ndarray, detected_faces: List[Tuple[int, int, int, int, float]]):
        """Process a frame for recognition using BlazeFace and InsightFace."""
        # Debug: Print frame info
        print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"[DEBUG] Frame sample pixel: {frame[0,0]}")
        # Ensure frame is RGB and uint8
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        frame_rgb = frame_rgb.astype(np.uint8)
        # Get InsightFace results for the full frame
        insightface_results = self.face_embedder.detect_and_encode_faces(frame_rgb)
        print(f"[DEBUG] InsightFace detected {len(insightface_results)} faces: {[f[0]['bbox'] for f in insightface_results]}")
        recognized_students = []
        for face_box in detected_faces:
            x, y, w, h, confidence = face_box
            blaze_bbox = [x, y, x + w, y + h]
            print(f"[DEBUG] BlazeFace bbox: {blaze_bbox}")
            def bbox_iou(b1, b2):
                xA = max(b1[0], b2[0])
                yA = max(b1[1], b2[1])
                xB = min(b1[2], b2[2])
                yB = min(b1[3], b2[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (b1[2] - b1[0]) * (b1[3] - b1[1])
                boxBArea = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                return iou
            best_iou = 0.0
            best_embedding = None
            for face_info, embedding in insightface_results:
                if embedding is not None:
                    iou = bbox_iou(face_info['bbox'], blaze_bbox)
                    print(f"[DEBUG] IOU between BlazeFace and InsightFace bbox: {iou:.2f}")
                    if iou > best_iou:
                        best_iou = iou
                        best_embedding = embedding
            if best_embedding is None or best_iou < 0.3:
                print(f"[ERROR] No matching embedding found for detected face at {blaze_bbox}. Saving frame for inspection.")
                import uuid
                fname = f"failed_insightface_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(fname, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                print(f"[ERROR] Saved problematic frame to {fname}")
                recognized_students.append({'student_id': None, 'confidence': 0.0, 'bbox': blaze_bbox})
                continue
            print(f"[DEBUG] Selected embedding for bbox {blaze_bbox} with IOU={best_iou:.2f}")
            best_student_id = None
            best_similarity = 0.0
            for student_id, enc_list in self.student_encodings.items():
                for stored_obj in enc_list:
                    stored_encoding = stored_obj.get('encoding')
                    if stored_encoding is None:
                        continue
                    # Skip if dims don't match
                    if stored_encoding.shape[0] != best_embedding.shape[0]:
                        continue
                    similarity = np.dot(best_embedding, stored_encoding) / (np.linalg.norm(best_embedding) * np.linalg.norm(stored_encoding))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_student_id = student_id
            threshold = self.threshold_var.get() if hasattr(self, 'threshold_var') else 0.6
            if best_similarity >= threshold:
                print(f"[DEBUG] Recognized student {best_student_id} with similarity {best_similarity:.4f}")
                recognized_students.append({'student_id': best_student_id, 'confidence': best_similarity, 'bbox': blaze_bbox})
            else:
                print(f"[DEBUG] No student matched above threshold for bbox {blaze_bbox}")
                recognized_students.append({'student_id': None, 'confidence': best_similarity, 'bbox': blaze_bbox})
        return recognized_students
