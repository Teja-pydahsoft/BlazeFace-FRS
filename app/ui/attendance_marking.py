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
from ..utils.camera_utils import CameraManager

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
        self.face_detector = BlazeFaceDetector(min_detection_confidence=0.3)  # Lower threshold for better detection
        self.pipeline = None
        self.face_embedder = None
        
        # Detect camera type for pipeline optimization
        self.camera_type = self._detect_camera_type(camera_source)
        
        # UI state
        self.is_running = False
        self.current_frame = None
        self.recognized_students = {}  # student_id -> (name, last_seen, confidence)
        self.attendance_marked = set()  # Set of student_ids already marked today
        self.student_names = {}  # Cache student names for quick lookup
        
        # Recognition consistency tracking
        self.last_recognition_result = None  # (student_id, confidence, timestamp)
        self.recognition_consistency_threshold = 0.05  # 5% variation allowed
        
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
        
        # Initialize face embedder
        self._initialize_face_embedder()
        
        # Load existing face encodings
        self._load_face_encodings()
        
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
        self.threshold_var = tk.DoubleVar(value=self.config.get('recognition_confidence', 0.80))
        threshold_scale = ttk.Scale(settings_frame, from_=0.6, to=1.0, 
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
        
        # Camera display
        self.camera_label = ttk.Label(camera_frame, text="Initializing camera...", 
                                     anchor=tk.CENTER, background='black', foreground='white')
        self.camera_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
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
        """Initialize face embedder with best available option"""
        try:
            # Try FaceNet first (most accurate)
            try:
                from ..core.facenet_embedder import FaceNetEmbedder
                facenet_model_path = self.config.get('facenet_model_path', 'models/facenet_keras.h5')
                self.face_embedder = FaceNetEmbedder(model_path=facenet_model_path)
                print("FaceNet embedder initialized successfully")
            except Exception as e:
                print(f"FaceNet embedder failed, trying Enhanced embedder: {e}")
                # Fallback to Enhanced embedder
                try:
                    from ..core.enhanced_face_embedder import EnhancedFaceEmbedder
                    self.face_embedder = EnhancedFaceEmbedder()
                    print("Enhanced face embedder initialized successfully")
                except Exception as e2:
                    print(f"Enhanced embedder failed, using Simple embedder: {e2}")
                    # Final fallback to Simple embedder
                    from ..core.simple_face_embedder import SimpleFaceEmbedder
                    self.face_embedder = SimpleFaceEmbedder()
                    print("Simple face embedder initialized successfully")
        except Exception as e:
            print(f"Error initializing face embedder: {e}")
            self.face_embedder = None
    
    def _load_face_encodings(self):
        """Load face encodings from database"""
        try:
            self.face_encodings = self.database_manager.get_face_encodings()
            self.student_encodings = {}  # student_id -> list of encodings
            self.student_names = {}  # student_id -> name
            
            for student_id, encoding, encoding_type in self.face_encodings:
                if student_id not in self.student_encodings:
                    self.student_encodings[student_id] = []
                    # Get student name and cache it
                    student_info = self.database_manager.get_student(student_id)
                    if student_info:
                        self.student_names[student_id] = student_info['name']
                    else:
                        self.student_names[student_id] = f"Student {student_id}"
                self.student_encodings[student_id].append(encoding)
            
            print(f"Loaded {len(self.face_encodings)} face encodings for {len(self.student_encodings)} students")
            
            # Debug: Show loaded encodings
            for student_id, encodings in self.student_encodings.items():
                print(f"  Student {student_id}: {len(encodings)} encodings")
                for i, encoding in enumerate(encodings):
                    print(f"    Encoding {i}: shape={encoding.shape}, norm={np.linalg.norm(encoding):.4f}")
                    print(f"    Encoding {i}: sample values={encoding[:5]}")
                
                # Check if encodings are different
                if len(encodings) > 1:
                    from app.core.simple_face_embedder import SimpleFaceEmbedder
                    embedder = SimpleFaceEmbedder()
                    is_same, similarity = embedder.compare_faces(encodings[0], encodings[1], 0.9)
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
                self.camera_label.config(text="Camera not available")
                self.status_label.config(text="Status: Camera not available", foreground='red')
        except Exception as e:
            print(f"Error starting camera preview: {e}")
            self.camera_label.config(text=f"Camera error: {str(e)}")
            self.status_label.config(text=f"Status: Error - {str(e)}", foreground='red')
    
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
                                print("No faces detected in current frame")
                            
                            frame = self._draw_enhanced_detections(frame, results)
                        
                        # Only do recognition processing at intervals
                        if current_time - self._last_recognition_time > 0.5:  # 0.5 second intervals
                            self._process_recognition_results(results)
                            self._last_recognition_time = current_time
                        else:
                            # Just draw existing detections without processing
                            frame = self._draw_enhanced_detections(frame, {'faces': [], 'embeddings': []})
                    elif self.is_running and self.camera_type == 'stream':
                        # Fallback: Direct face detection for NVR cameras if pipeline fails
                        if self._frame_skip_counter % 5 == 0:  # Even less frequent for fallback
                            faces = self.face_detector.detect_faces(frame)
                            if faces:
                                print(f"Direct NVR detection: Found {len(faces)} faces")
                            
                            # Draw faces directly
                            frame = self.face_detector.draw_faces(frame, faces)
                    
                    # Resize frame for display
                    display_frame = cv2.resize(frame, (500, 400))
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update display
                    self.camera_label.config(image=frame_tk, text="")
                    self.camera_label.image = frame_tk
                    
                    self.current_frame = frame
                else:
                    # Only show "No camera feed" if we've been trying for a while
                    if not hasattr(self, '_no_feed_count'):
                        self._no_feed_count = 0
                    self._no_feed_count += 1
                    
                    if self._no_feed_count > 15:  # Increased threshold for more stability
                        self.camera_label.config(text="No camera feed")
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
            self.camera_label.config(text=f"Camera error: {str(e)}")
            self.status_label.config(text=f"Status: Error - {str(e)}", foreground='red')
            self.dialog.after(1000, self._update_camera_preview)
    
    def _draw_enhanced_detections(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Draw enhanced detections with recognition status"""
        try:
            faces = results.get('faces', [])
            embeddings = results.get('embeddings', [])
            
            if not faces:
                return frame
            
            result_frame = frame.copy()
            
            # Process each detected face
            for i, face_box in enumerate(faces):
                x, y, w, h, confidence = face_box
                
                # Skip faces with low detection confidence
                min_face_confidence = self.face_confidence_var.get()
                if confidence < min_face_confidence:
                    print(f"Face {i}: Skipping - low detection confidence {confidence:.2f} < {min_face_confidence:.2f}")
                    # Draw red box for low confidence faces but still show them
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(result_frame, f"Face: {confidence:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(result_frame, f"Low Confidence (<{min_face_confidence:.1f})", 
                               (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # Don't continue - still process as unknown face
                
                # Get embedding for this face
                embedding = None
                if i < len(embeddings) and embeddings[i] is not None:
                    embedding = embeddings[i]
                
                # Find best match
                student_name = "Unknown Face"
                color = (0, 0, 255)  # Red for unknown
                status_text = "Unknown Face"
                
                if embedding is not None:
                    print(f"Face {i}: Processing recognition with confidence {confidence:.2f}")
                    print(f"Face {i}: Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
                    print(f"Face {i}: Available students: {list(self.student_encodings.keys())}")
                    best_match = self._find_best_match(embedding)
                    if best_match:
                        student_id, match_confidence = best_match
                        student_name = self.student_names.get(student_id, f"Student {student_id}")
                        color = (0, 255, 0)  # Green for recognized
                        status_text = f"Recognized: {student_name}"
                        print(f"Face {i}: MATCHED {student_name} with confidence {match_confidence:.4f}")
                    else:
                        print(f"Face {i}: NO MATCH found - will show as Unknown Face")
                        color = (0, 0, 255)  # Red for unknown
                        status_text = "Unknown Face - Not Registered"
                else:
                    print(f"Face {i}: No embedding available - will show as Unknown Face")
                    color = (0, 0, 255)  # Red for unknown
                    status_text = "Unknown Face - No Embedding"
                
                # Draw face bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence
                cv2.putText(result_frame, f"Face: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw recognition status
                cv2.putText(result_frame, status_text, 
                           (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw student name if recognized
                if "Recognized:" in status_text:
                    cv2.putText(result_frame, f"Student: {student_name}", 
                               (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return result_frame
            
        except Exception as e:
            print(f"Error drawing enhanced detections: {e}")
            return frame
    
    def _process_recognition_results(self, results: Dict[str, Any]):
        """Process recognition results and mark attendance"""
        try:
            faces = results.get('faces', [])
            embeddings = results.get('embeddings', [])
            
            # Debug information
            if faces:
                print(f"Processing {len(faces)} faces, {len(embeddings)} embeddings")
                for i, face in enumerate(faces):
                    x, y, w, h, conf = face
                    print(f"  Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
            else:
                print("No faces detected in current frame")
            
            if not faces:
                return
            
            # Process each detected face
            for i, face in enumerate(faces):
                x, y, w, h, detection_confidence = face
                
                # Skip faces with low detection confidence for recognition processing
                min_detection_confidence = self.face_confidence_var.get()
                if detection_confidence < min_detection_confidence:
                    print(f"Face {i}: Skipping recognition - low detection confidence {detection_confidence:.2f} < {min_detection_confidence:.2f}")
                    # Still log as unknown face for display purposes
                    print(f"Face {i}: Will be displayed as Unknown Face (low confidence)")
                    continue
                
                # Get embedding for this face (if available)
                embedding = None
                if i < len(embeddings) and embeddings[i] is not None:
                    embedding = embeddings[i]
                
                if embedding is None:
                    print(f"Face {i}: No embedding available - showing as unknown")
                    continue
                
                print(f"Face {i}: Processing recognition with detection confidence {detection_confidence:.2f}")
                
                # Find best match using face embedding similarity
                best_match = self._find_best_match(embedding)
                
                if best_match:
                    student_id, recognition_confidence = best_match
                    print(f"Face {i}: Matched student {student_id} with recognition confidence {recognition_confidence:.4f}")
                    
                    # Update recognized students
                    student_info = self.database_manager.get_student(student_id)
                    if student_info:
                        self.recognized_students[student_id] = (
                            student_info['name'],
                            time.time(),
                            recognition_confidence  # Store recognition confidence, not detection confidence
                        )
                        
                        # Auto-mark attendance if enabled
                        if self.auto_mark_var.get():
                            print(f"Auto-marking attendance for {student_id} with recognition confidence {recognition_confidence:.4f}")
                            self._mark_attendance(student_id, recognition_confidence)
                        else:
                            print(f"Auto-mark disabled - would mark {student_id} with recognition confidence {recognition_confidence:.4f}")
                else:
                    print(f"Face {i}: No match found - face will be displayed as UNKNOWN")
            
            # Update recognition info display
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
            
            # Get recognition threshold from config or UI
            ui_threshold = self.threshold_var.get()
            config_threshold = self.config.get('recognition_confidence', 0.80)
            
            # Use balanced threshold for better performance - CONSISTENT THRESHOLD
            recognition_threshold = max(ui_threshold, config_threshold, 0.75)  # Lowered minimum to 0.75
            
            print(f"Using UI threshold: {ui_threshold:.2f}, config threshold: {config_threshold:.2f}, final threshold: {recognition_threshold:.2f}")
            print(f"Comparing against {len(self.student_encodings)} students")
            
            best_similarity = 0.0
            best_student_id = None
            all_similarities = []  # Track all similarities for debugging
            
            for student_id, encodings in self.student_encodings.items():
                print(f"Checking student {student_id} with {len(encodings)} encodings")
                student_max_similarity = 0.0
                
                for i, encoding in enumerate(encodings):
                    # Use very low threshold to get actual similarity scores
                    is_same, similarity = self.face_embedder.compare_faces(query_embedding, encoding, 0.01)
                    all_similarities.append((student_id, i, similarity))
                    # Track the highest similarity for this student
                    if similarity > student_max_similarity:
                        student_max_similarity = similarity
                    print(f"  Encoding {i}: similarity={similarity:.4f}")
                
                print(f"  Student {student_id} max similarity: {student_max_similarity:.4f}")
                
                # Track the best match regardless of threshold (threshold check happens later)
                if student_max_similarity > best_similarity:
                    best_similarity = student_max_similarity
                    best_student_id = student_id
                    print(f"  -> New best match: {student_id} with similarity {student_max_similarity:.4f}")
            
            # Debug: Show all similarities
            print("All similarities:")
            for student_id, enc_idx, sim in sorted(all_similarities, key=lambda x: x[2], reverse=True)[:5]:
                print(f"  {student_id}[{enc_idx}]: {sim:.4f}")
            
            print(f"Final result: best_similarity={best_similarity:.4f}, best_student_id={best_student_id}")
            
            # QUALITY CHECK: Accept matches above threshold, or best match if no one meets threshold
            if best_similarity >= recognition_threshold:
                # Additional check: Ensure this is a clear, unambiguous match
                second_best = 0.0
                for student_id, enc_idx, sim in sorted(all_similarities, key=lambda x: x[2], reverse=True)[1:3]:
                    if sim > second_best:
                        second_best = sim
                
                # Calculate gap between best and second best
                gap = best_similarity - second_best
                min_gap = self.config.get('min_confidence_gap', 0.01)  # Reduced from 0.02 to 0.01
                
                # BALANCED LOGIC: Require good confidence with reasonable gap
                if best_similarity >= recognition_threshold and gap >= min_gap:
                    # Additional consistency check: Compare with last recognition result
                    current_time = time.time()
                    if self.last_recognition_result:
                        last_student_id, last_confidence, last_time = self.last_recognition_result
                        time_diff = current_time - last_time
                        
                        # If recognition happened recently (within 1 second) and results are inconsistent
                        if time_diff < 1.0:  # Reduced from 2.0 to 1.0 seconds
                            if last_student_id != best_student_id:
                                # Only reject if confidence difference is very large (similar students can have close scores)
                                confidence_diff = abs(best_similarity - last_confidence)
                                if confidence_diff > 0.1:  # Only reject if difference > 0.1
                                    print(f"⚠️  INCONSISTENT RECOGNITION: Last result was {last_student_id} ({last_confidence:.4f}), now {best_student_id} ({best_similarity:.4f})")
                                    print(f"⚠️  REJECTING INCONSISTENT MATCH - This face will be marked as UNKNOWN")
                                    return None
                                else:
                                    print(f"ℹ️  SIMILAR STUDENTS: Switching from {last_student_id} to {best_student_id} (diff: {confidence_diff:.4f})")
                            elif abs(best_similarity - last_confidence) > self.recognition_consistency_threshold:
                                print(f"⚠️  CONFIDENCE VARIATION: Last {last_confidence:.4f}, now {best_similarity:.4f} (diff: {abs(best_similarity - last_confidence):.4f})")
                                print(f"⚠️  REJECTING VARIABLE MATCH - This face will be marked as UNKNOWN")
                                return None
                    
                    # Store this recognition result for consistency checking
                    self.last_recognition_result = (best_student_id, best_similarity, current_time)
                    
                    print(f"✓ CLEAR MATCH FOUND: {best_student_id} with similarity {best_similarity:.4f} (gap: {gap:.4f})")
                    return best_student_id, best_similarity
                else:
                    print(f"✗ INSUFFICIENT CONFIDENCE: {best_similarity:.4f} < {recognition_threshold:.2f} OR gap {gap:.4f} < {min_gap:.3f}")
                    print(f"✗ REJECTING MATCH - This face will be marked as UNKNOWN")
                    return None
            else:
                # Fallback: If best match is below threshold but still reasonable, accept it with lower confidence
                if best_similarity >= 0.70:  # Lower threshold for fallback
                    print(f"⚠️  FALLBACK MATCH: Best similarity {best_similarity:.4f} below threshold {recognition_threshold:.2f} but above 0.70")
                    print(f"⚠️  ACCEPTING FALLBACK MATCH: {best_student_id} with similarity {best_similarity:.4f}")
                    return best_student_id, best_similarity
                else:
                    print(f"✗ NO MATCH: Best similarity {best_similarity:.4f} below recognition threshold {recognition_threshold:.2f}")
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
            
            # Initialize pipeline with camera type
            pipeline_config = self.config.copy()
            pipeline_config['camera_type'] = self.camera_type
            self.pipeline = DualPipeline(pipeline_config)
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
                    for stored_encoding in encodings:
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
