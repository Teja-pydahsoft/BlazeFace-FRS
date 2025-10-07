"""
Enhanced Student Registration Dialog for BlazeFace-FRS System
Handles both camera capture and high-quality image upload for face registration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
import logging

from ..core.simple_face_embedder import SimpleFaceEmbedder
from ..core.standard_face_embedder import StandardFaceEmbedder
from ..core.insightface_embedder import InsightFaceEmbedder
from ..core.blazeface_detector import BlazeFaceDetector
from ..core.database import DatabaseManager
from ..utils.camera_utils import CameraManager
from ..utils.encoding_quality_checker import EncodingQualityChecker
from ..core.faiss_index import FaissIndex

class EnhancedRegistrationDialog:
    def __init__(self, parent, database_manager: DatabaseManager, config: Dict[str, Any], 
                 existing_camera_manager: CameraManager = None):
        """
        Initialize enhanced registration dialog
        
        Args:
            parent: Parent window
            database_manager: Database manager instance
            config: Application configuration
            existing_camera_manager: Existing camera manager to reuse (optional)
        """
        self.parent = parent
        self.database_manager = database_manager
        self.config = config
        
        # Initialize components
        self.face_detector = BlazeFaceDetector(
            min_detection_confidence=config.get('detection_confidence', 0.7)
        )
        
        # Initialize embedders
        self.embedders = {
            'simple': SimpleFaceEmbedder(),
            'standard': StandardFaceEmbedder(),
            'insightface': InsightFaceEmbedder()
        }
        # Default to InsightFace (512-d) for registration to build canonical gallery
        self.current_embedder = 'insightface'
        self.embedder_var = tk.StringVar(value="insightface") # Set default to insightface
        
        # Use existing camera manager or create new one
        if existing_camera_manager and existing_camera_manager.is_camera_available():
            self.camera_manager = existing_camera_manager
            self.using_shared_camera = True
            print("Using shared camera manager")
        else:
            self.camera_manager = CameraManager(config.get('camera_index', 0))
            self.using_shared_camera = False
            print("Created new camera manager for registration")
            
        self.quality_checker = EncodingQualityChecker()
        
        # UI state
        self.is_capturing = False
        self.captured_faces = []
        self.current_frame = None
        self.camera_running = True
        self.registration_mode = "camera"  # "camera" or "image"
        self.uploaded_images = []
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Enhanced Student Registration")
        self.dialog.geometry("1000x700")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Setup UI
        self._setup_ui()
        
        # Handle dialog close event
        self.dialog.protocol("WM_DELETE_WINDOW", self._cancel)
        
        # Ensure dialog is fully initialized before starting camera
        self.dialog.update_idletasks()
        
        # Start camera preview if in camera mode
        if self.registration_mode == "camera":
            self._start_camera_preview()
        
        # Center dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Setup the enhanced registration dialog UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Form and Controls
        self._setup_form_panel(main_frame)
        
        # Right panel - Camera/Image and Face Capture
        self._setup_capture_panel(main_frame)
        
        # Bottom panel - Buttons
        self._setup_button_panel(main_frame)
    
    def _setup_form_panel(self, parent):
        """Setup student information form and controls"""
        form_frame = ttk.LabelFrame(parent, text="Student Information & Settings", padding="10")
        form_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Registration mode selection
        mode_frame = ttk.LabelFrame(form_frame, text="Registration Mode", padding="5")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="camera")
        ttk.Radiobutton(mode_frame, text="Camera Capture", variable=self.mode_var, 
                       value="camera", command=self._on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="High-Quality Images", variable=self.mode_var, 
                       value="image", command=self._on_mode_change).pack(anchor=tk.W)
        
        # Face embedder selection
        embedder_frame = ttk.LabelFrame(form_frame, text="Face Encoder", padding="5")
        embedder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(embedder_frame, text="Encoder Type:").pack(anchor=tk.W)
        self.embedder_var = tk.StringVar(value="standard")
        embedder_combo = ttk.Combobox(embedder_frame, textvariable=self.embedder_var,
                                     values=list(self.embedders.keys()), state="readonly")
        embedder_combo.pack(fill=tk.X, pady=(5, 0))
        embedder_combo.bind('<<ComboboxSelected>>', self._on_embedder_change)
        
        # Student information form
        info_frame = ttk.LabelFrame(form_frame, text="Student Details", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Student ID
        ttk.Label(info_frame, text="Student ID *:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.student_id_var = tk.StringVar()
        self.student_id_entry = ttk.Entry(info_frame, textvariable=self.student_id_var, width=20)
        self.student_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Name
        ttk.Label(info_frame, text="Full Name *:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(info_frame, textvariable=self.name_var, width=20)
        self.name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Email
        ttk.Label(info_frame, text="Email:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.email_var = tk.StringVar()
        self.email_entry = ttk.Entry(info_frame, textvariable=self.email_var, width=20)
        self.email_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Phone
        ttk.Label(info_frame, text="Phone:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.phone_var = tk.StringVar()
        self.phone_entry = ttk.Entry(info_frame, textvariable=self.phone_var, width=20)
        self.phone_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Department
        ttk.Label(info_frame, text="Department:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.department_var = tk.StringVar()
        self.department_combo = ttk.Combobox(info_frame, textvariable=self.department_var, 
                                           values=["Computer Science", "Engineering", "Business", "Arts", "Science"])
        self.department_combo.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Year
        ttk.Label(info_frame, text="Year:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.year_var = tk.StringVar()
        self.year_combo = ttk.Combobox(info_frame, textvariable=self.year_var,
                                     values=["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduate"])
        self.year_combo.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Configure column weights
        info_frame.columnconfigure(1, weight=1)
        
        # Quality settings
        quality_frame = ttk.LabelFrame(form_frame, text="Quality Settings", padding="5")
        quality_frame.pack(fill=tk.X)
        
        ttk.Label(quality_frame, text="Min Quality Score:").pack(anchor=tk.W)
        self.quality_var = tk.DoubleVar(value=0.7)
        quality_scale = ttk.Scale(quality_frame, from_=0.3, to=1.0, 
                                variable=self.quality_var, orient=tk.HORIZONTAL)
        quality_scale.pack(fill=tk.X, pady=(5, 0))
        
        self.quality_label = ttk.Label(quality_frame, text="0.7")
        self.quality_label.pack()
        quality_scale.config(command=lambda v: self.quality_label.config(text=f"{float(v):.2f}"))
    
    def _setup_capture_panel(self, parent):
        """Setup camera/image capture panel"""
        capture_frame = ttk.LabelFrame(parent, text="Face Capture", padding="10")
        capture_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera/Image display
        self.display_canvas = tk.Canvas(capture_frame, width=500, height=400, 
                                       background='black', highlightthickness=1,
                                       relief='sunken', bd=2)
        self.display_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Add initial text overlay
        self.display_canvas.create_text(250, 200, text="Select registration mode above", 
                                       fill='white', font=('Arial', 14), tags='status_text')
        
        # Status label
        self.status_label = ttk.Label(capture_frame, text="Status: Ready", 
                                     font=('Arial', 9), foreground='blue')
        self.status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Mode-specific controls
        self.controls_frame = ttk.Frame(capture_frame)
        self.controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera controls (initially shown)
        self._setup_camera_controls()
        
        # Image upload controls (initially hidden)
        self._setup_image_controls()
        
        # Captured faces display
        faces_frame = ttk.LabelFrame(capture_frame, text="Captured Faces", padding="5")
        faces_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame for captured faces
        canvas = tk.Canvas(faces_frame, height=120)
        scrollbar = ttk.Scrollbar(faces_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _setup_camera_controls(self):
        """Setup camera capture controls"""
        self.camera_controls = ttk.Frame(self.controls_frame)
        self.camera_controls.pack(fill=tk.X)
        
        self.capture_btn = ttk.Button(self.camera_controls, text="Capture Face", 
                                     command=self._capture_face, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_btn = ttk.Button(self.camera_controls, text="Clear Faces", 
                                   command=self._clear_faces)
        self.clear_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        self.debug_btn = ttk.Button(self.camera_controls, text="Debug Camera", 
                                   command=self._debug_camera)
        self.debug_btn.pack(side=tk.LEFT, padx=(5, 0))
    
    def _setup_image_controls(self):
        """Setup image upload controls"""
        self.image_controls = ttk.Frame(self.controls_frame)
        # Initially hidden
        
        # Image upload section
        upload_frame = ttk.LabelFrame(self.image_controls, text="Upload High-Quality Images", padding="5")
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(upload_frame, text="Select Images", 
                  command=self._select_images).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(upload_frame, text="Process Images", 
                  command=self._process_uploaded_images, state=tk.DISABLED).pack(side=tk.LEFT, padx=(0, 10))
        
        self.process_btn = ttk.Button(upload_frame, text="Process Images", 
                                     command=self._process_uploaded_images, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Image preview
        preview_frame = ttk.LabelFrame(self.image_controls, text="Image Preview", padding="5")
        preview_frame.pack(fill=tk.X)
        
        self.image_listbox = tk.Listbox(preview_frame, height=4)
        self.image_listbox.pack(fill=tk.X, pady=(0, 5))
        self.image_listbox.bind('<<ListboxSelect>>', self._preview_selected_image)
        
        # Instructions
        instructions = ttk.Label(self.image_controls, 
                               text="Instructions:\n1. Select high-quality JPEG/PNG images\n2. Images should show clear, well-lit faces\n3. Multiple angles improve recognition\n4. Click 'Process Images' to extract faces",
                               justify=tk.LEFT, font=('Arial', 9))
        instructions.pack(fill=tk.X, pady=(10, 0))
    
    def _setup_button_panel(self, parent):
        """Setup button panel"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(button_frame, text="Register Student", 
                  command=self._register_student).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", 
                  command=self._cancel).pack(side=tk.RIGHT)
    
    def _on_mode_change(self):
        """Handle registration mode change"""
        self.registration_mode = self.mode_var.get()
        
        if self.registration_mode == "camera":
            # Show camera controls, hide image controls
            self.camera_controls.pack(fill=tk.X)
            self.image_controls.pack_forget()
            
            # Start camera if not running
            if not self.camera_running:
                self._start_camera_preview()
        else:
            # Hide camera controls, show image controls
            self.camera_controls.pack_forget()
            self.image_controls.pack(fill=tk.X)
            
            # Stop camera
            self.camera_running = False
            self.display_canvas.delete('all')
            self.display_canvas.create_text(250, 200, text="Upload high-quality images below", 
                                           fill='white', font=('Arial', 14), tags='status_text')
            self.status_label.config(text="Status: Ready for image upload", foreground='blue')
    
    def _on_embedder_change(self, event=None):
        """Handle embedder selection change"""
        self.current_embedder = self.embedder_var.get()
        print(f"Switched to embedder: {self.current_embedder}")
    
    def _start_camera_preview(self):
        """Start camera preview"""
        try:
            print(f"Starting camera preview - Using shared camera: {self.using_shared_camera}")
            
            if self.camera_manager.is_camera_available():
                self.capture_btn.config(state=tk.NORMAL)
                self.display_canvas.delete('all')
                self.display_canvas.create_text(250, 200, text="Initializing camera...", 
                                              fill='white', font=('Arial', 12), tags='status_text')
                self.status_label.config(text="Status: Initializing camera...", foreground='blue')
                
                # Start camera loop in separate thread
                self.camera_running = True
                self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
                self.camera_thread.start()
            else:
                self.display_canvas.delete('all')
                self.display_canvas.create_text(250, 200, text="Camera not available", 
                                              fill='red', font=('Arial', 12), tags='status_text')
                self.status_label.config(text="Status: Camera not available", foreground='red')
        except Exception as e:
            print(f"Error starting camera preview: {e}")
            self.display_canvas.delete('all')
            self.display_canvas.create_text(250, 200, text=f"Camera error: {str(e)}", 
                                          fill='red', font=('Arial', 12), tags='status_text')
            self.status_label.config(text=f"Status: Error - {str(e)}", foreground='red')
    
    def _camera_loop(self):
        """Camera preview loop"""
        try:
            while self.camera_running:
                # Check if dialog still exists
                try:
                    if not self.dialog.winfo_exists():
                        break
                except tk.TclError:
                    break
                
                if not self.camera_manager.is_camera_available():
                    break
                
                ret, frame = self.camera_manager.get_frame()
                if not ret:
                    continue
                
                # Get frame dimensions
                h, w = frame.shape[:2]
                frame_area = h * w
                
                # Convert to RGB for InsightFace
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use InsightFace's detector for consistent detection
                display_frame = frame.copy()
                face_detected = False
                guidance_text = "Position your face in the camera"
                guidance_color = (255, 255, 255)  # White
                
                try:
                    if self.embedders['insightface'].app is not None:
                        faces = self.embedders['insightface'].app.get(rgb_frame)
                        if faces:
                            for face in faces:
                                bbox = face.bbox.astype(int)
                                x1, y1, x2, y2 = bbox
                                w, h = x2 - x1, y2 - y1
                                face_area = w * h
                                area_ratio = face_area / frame_area
                                
                                # Draw face rectangle
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Provide specific guidance based on face size and position
                                if area_ratio < 0.05:  # Face too small
                                    guidance_text = "Move closer to the camera"
                                    guidance_color = (255, 165, 0)  # Orange
                                elif area_ratio > 0.4:  # Face too large
                                    guidance_text = "Move back from the camera"
                                    guidance_color = (255, 165, 0)  # Orange
                                elif x1 < 50 or y1 < 50 or x2 > w-50 or y2 > h-50:
                                    guidance_text = "Center your face in the frame"
                                    guidance_color = (255, 165, 0)  # Orange
                                else:
                                    guidance_text = "Perfect! Ready to capture"
                                    guidance_color = (0, 255, 0)  # Green
                                    face_detected = True
                                
                                # Display confidence
                                conf_text = f"Quality: {face.det_score:.2f}"
                                cv2.putText(display_frame, conf_text, 
                                          (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.7, (0, 255, 0), 2)
                        else:
                            guidance_text = "No face detected - Look directly at camera"
                            guidance_color = (0, 0, 255)  # Red
                except Exception as e:
                    # Fallback to basic detection if InsightFace fails
                    faces = self.face_detector.detect_faces(frame)
                    for face in faces:
                        x, y, w, h, confidence = face
                        if confidence > 0.7:
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), 
                                        (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Face: {confidence:.2f}", 
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (0, 255, 0), 2)
                            face_detected = True
                
                # Add guidance text at the top of the frame
                cv2.putText(display_frame, guidance_text, 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, guidance_color, 2)
                
                # Convert to PhotoImage and update display
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(display_frame, (500, 400))
                
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image)
                
                # Update UI in main thread
                self.dialog.after(0, self._update_camera_display, photo, face_detected, guidance_text, self._bgr_to_hex(guidance_color))
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Camera loop error: {e}")
            self.dialog.after(0, lambda: self.status_label.config(text="Camera error", foreground='red'))
    
    def _bgr_to_hex(self, color_bgr: Tuple[int, int, int]) -> str:
        """Converts a BGR color tuple to a Tkinter-compatible hex string."""
        # OpenCV uses BGR, Tkinter uses RGB hex. The guidance color is created as RGB, so we just format it.
        r, g, b = color_bgr
        return f'#{r:02x}{g:02x}{b:02x}'

    def _update_camera_display(self, photo, face_detected, guidance_text, guidance_color_hex):
        """Update camera display in main thread"""
        try:
            # Check if dialog still exists
            if not self.dialog.winfo_exists():
                return
                
            self.display_canvas.delete('all')
            self.display_canvas.create_image(250, 200, image=photo, anchor=tk.CENTER)
            self.display_canvas.image = photo  # Keep a reference
            
            # Update status label with detailed guidance
            self.status_label.config(text=f"Status: {guidance_text}", foreground=guidance_color_hex)
                
        except tk.TclError:
            # Dialog was closed, stop camera loop
            return
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
    def _is_image_bright_and_clear(self, img: np.ndarray) -> bool:
        """Check if the image is well-lit and not over/underexposed."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img
        mean = np.mean(gray)
        std = np.std(gray)
        # Acceptable brightness: mean between 60 and 200, std above 30
        return 60 < mean < 200 and std > 30

    def _capture_face(self):
        """Capture face from current camera frame"""
        try:
            if not self.camera_manager.is_camera_available():
                messagebox.showwarning("Warning", "Camera not available")
                return
            
            ret, frame = self.camera_manager.get_frame()
            if not ret:
                messagebox.showwarning("Warning", "No camera feed available")
                return
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_area = rgb_frame.shape[0] * rgb_frame.shape[1]
            
            if self.current_embedder == 'insightface':
                try:
                    embedder = self.embedders['insightface']
                    if embedder.app is None:
                        messagebox.showerror("Error", "InsightFace embedder not initialized properly")
                        return
                    faces = embedder.app.get(rgb_frame)
                    if not faces:
                        messagebox.showwarning("Warning", 
                            "No face detected. Please ensure:\n" +
                            "1. You are looking directly at the camera\n" +
                            "2. Your face is well-lit\n" +
                            "3. You are at a good distance (not too close/far)")
                        return
                    best_face = max(faces, key=lambda f: f.det_score)
                    bbox = best_face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    face_area = w * h
                    area_ratio = face_area / frame_area
                    # Area ratio check: face should be 10%-40% of frame
                    if area_ratio < 0.10:
                        messagebox.showwarning("Warning", "Face too small. Move closer to the camera.")
                        return
                    if area_ratio > 0.40:
                        messagebox.showwarning("Warning", "Face too large. Move back from the camera.")
                        return
                    # Centering check
                    margin = 40
                    if x1 < margin or y1 < margin or x2 > rgb_frame.shape[1] - margin or y2 > rgb_frame.shape[0] - margin:
                        messagebox.showwarning("Warning", "Center your face in the frame.")
                        return
                    # Brightness/contrast check
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    if not self._is_image_bright_and_clear(face_crop):
                        messagebox.showwarning("Warning", "Face region is too dark, too bright, or blurry. Adjust lighting and try again.")
                        return
                    face_region = rgb_frame
                except Exception as e:
                    messagebox.showerror("Error", f"InsightFace detection failed: {str(e)}")
                    return
            else:
                faces = self.face_detector.detect_faces(frame)
                if not faces:
                    messagebox.showwarning("Warning", "No face detected in current frame")
                    return
                best_face = max(faces, key=lambda f: f[4])
                x, y, w, h, confidence = best_face
                face_area = w * h
                area_ratio = face_area / frame_area
                if area_ratio < 0.10:
                    messagebox.showwarning("Warning", "Face too small. Move closer to the camera.")
                    return
                if area_ratio > 0.40:
                    messagebox.showwarning("Warning", "Face too large. Move back from the camera.")
                    return
                margin = 40
                if x < margin or y < margin or x + w > frame.shape[1] - margin or y + h > frame.shape[0] - margin:
                    messagebox.showwarning("Warning", "Center your face in the frame.")
                    return
                face_region = self.face_detector.extract_face_region(frame, (x, y, w, h))
                if face_region is not None and face_region.size > 0:
                    if not self._is_image_bright_and_clear(face_region):
                        messagebox.showwarning("Warning", "Face region is too dark, too bright, or blurry. Adjust lighting and try again.")
                        return
            
            # Get face embedding
            embedder = self.embedders[self.current_embedder]
            
            # If current embedder is InsightFace, try with a slightly larger region if initial detection fails
            if self.current_embedder == 'insightface':
                embedding = embedder.get_embedding(face_region)
                if embedding is None:
                    # Try expanding the face region slightly and re-embedding
                    expanded_x = max(0, x - int(w * 0.1))
                    expanded_y = max(0, y - int(h * 0.1))
                    expanded_w = min(frame.shape[1] - expanded_x, int(w * 1.2))
                    expanded_h = min(frame.shape[0] - expanded_y, int(h * 1.2))
                    expanded_face_region = self.face_detector.extract_face_region(frame, (expanded_x, expanded_y, expanded_w, expanded_h))
                    if expanded_face_region is not None and expanded_face_region.size > 0:
                        embedding = embedder.get_embedding(expanded_face_region)
            else:
                embedding = embedder.get_embedding(face_region)

            if embedding is None:
                messagebox.showwarning("Warning", f"Failed to get embedding with {self.current_embedder} embedder. Please ensure your face is clear and well-lit.")
                return
            
            # Check encoding quality
            student_id = self.student_id_var.get().strip()
            
            # Get existing encodings for quality check
            existing_encodings = []
            other_encodings = []
            if student_id:
                existing_encs = self.database_manager.get_face_encodings(student_id)
                existing_encodings = [enc for _, enc, _, _ in existing_encs]
            
            all_encs = self.database_manager.get_face_encodings()
            for sid, enc, _, _ in all_encs:
                if sid != student_id:
                    other_encodings.append(enc)
            
            # Check quality
            is_acceptable, reason, quality_metrics = self.quality_checker.check_new_encoding_quality(
                embedding, existing_encodings, other_encodings, student_id
            )
            
            # Add to captured faces list
            if is_acceptable:
                self.captured_faces.append({
                    "image": face_region,
                    "embedding": embedding,
                    "quality": quality_metrics,
                    "embedder": self.current_embedder
                })
                self._update_captured_faces_display()
            else:
                messagebox.showwarning("Quality Warning", 
                                     f"Face not added due to low quality: {reason}\n"
                                     f"Details: {quality_metrics}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during face capture: {str(e)}")
    
    def _update_captured_faces_display(self):
        """Update the display of captured faces"""
        # Clear existing faces
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Add new faces
        for i, face_data in enumerate(self.captured_faces):
            face_img = face_data['image']
            if face_img.shape[2] == 3:  # Check if it's a color image
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(face_img)
            img.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(img)
            
            face_label = ttk.Label(self.scrollable_frame, image=photo, 
                                 text=f"Face {i+1}", compound=tk.TOP)
            face_label.image = photo
            face_label.pack(side=tk.LEFT, padx=5, pady=5)
    
    def _clear_faces(self):
        """Clear all captured faces"""
        self.captured_faces = []
        self._update_captured_faces_display()
    
    def _debug_camera(self):
        """Show camera debug information"""
        if self.camera_manager.is_camera_available():
            info = self.camera_manager.get_camera_info()
            messagebox.showinfo("Camera Debug Info", json.dumps(info, indent=2))
        else:
            messagebox.showwarning("Warning", "Camera not available")
    
    def _select_images(self):
        """Select high-quality images for registration"""
        filepaths = filedialog.askopenfilenames(
            title="Select High-Quality Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if filepaths:
            self.uploaded_images = list(filepaths)
            self.image_listbox.delete(0, tk.END)
            for path in self.uploaded_images:
                self.image_listbox.insert(tk.END, os.path.basename(path))
            self.process_btn.config(state=tk.NORMAL)
    
    def _preview_selected_image(self, event=None):
        """Preview the selected image in the listbox"""
        selection = self.image_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        image_path = self.uploaded_images[index]
        
        try:
            img = Image.open(image_path)
            img.thumbnail((500, 400))
            photo = ImageTk.PhotoImage(img)
            
            self.display_canvas.delete('all')
            self.display_canvas.create_image(250, 200, image=photo, anchor=tk.CENTER)
            self.display_canvas.image = photo
        except Exception as e:
            self.display_canvas.delete('all')
            self.display_canvas.create_text(250, 200, text=f"Error previewing image:\n{e}",
                                           fill='red', font=('Arial', 12))
    
    def _process_uploaded_images(self):
        """Process uploaded images to extract face embeddings"""
        if not self.uploaded_images:
            messagebox.showwarning("Warning", "No images selected")
            return
        
        self.captured_faces = []
        
        for image_path in self.uploaded_images:
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Use InsightFace for processing uploaded images
                embedder = self.embedders['insightface']
                results = embedder.detect_and_encode_faces(img)
                
                for x, y, w, h, confidence, embedding in results:
                    face_region = self.face_detector.extract_face_region(img, (x, y, w, h))
                    
                    self.captured_faces.append({
                        "image": face_region,
                        "embedding": embedding,
                        "quality": {"confidence": confidence},
                        "embedder": "insightface"
                    })
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        self._update_captured_faces_display()
        messagebox.showinfo("Info", f"Processed {len(self.uploaded_images)} images, found {len(self.captured_faces)} faces.")
    
    def _register_student(self):
        """Register student with captured faces"""
        student_id = self.student_id_var.get().strip()
        name = self.name_var.get().strip()
        
        if not student_id or not name:
            messagebox.showerror("Error", "Student ID and Full Name are required")
            return
        
        if not self.captured_faces:
            messagebox.showerror("Error", "No faces captured for registration")
            return
        
        try:
            # Add student to database
            self.database_manager.add_student(
                student_id=student_id,
                name=name,
                email=self.email_var.get().strip(),
                phone=self.phone_var.get().strip(),
                department=self.department_var.get(),
                year=self.year_var.get()
            )
            
            # Add face encodings
            encodings = [face['embedding'] for face in self.captured_faces]
            embedder_types = [face['embedder'] for face in self.captured_faces]
            
            # For simplicity, assume all captured faces have same embedder type
            # In a more complex scenario, this might need to be handled per-encoding
            embedder_type = embedder_types[0] if embedder_types else 'unknown'
            
            # Save one of the captured images for reference
            # In a real application, you might want to save all or the best one
            image_path = None
            if self.captured_faces:
                try:
                    save_dir = "face_data"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Use the first captured face image
                    face_img_to_save = self.captured_faces[0]['image']
                    if face_img_to_save.shape[2] == 3:
                        face_img_to_save = cv2.cvtColor(face_img_to_save, cv2.COLOR_RGB2BGR)
                    
                    filename = f"{student_id}_{int(time.time() * 1000)}.jpg"
                    image_path = os.path.join(save_dir, filename)
                    cv2.imwrite(image_path, face_img_to_save)
                except Exception as e:
                    print(f"Error saving face image: {e}")
            
            self.database_manager.add_face_encoding(
                student_id=student_id,
                encoding=encodings[0],  # For now, add first encoding
                encoding_type=embedder_type,
                image_path=image_path
            )
            
            # Attempt to update FAISS index
            try:
                faiss_path = os.path.join('data', 'faiss')
                fi = FaissIndex()
                if os.path.exists(faiss_path + '.meta.json'):
                    fi.load(faiss_path)
                
                new_embeddings = np.array(encodings, dtype=np.float32)
                new_names = [student_id] * len(encodings)
                
                fi.add(new_embeddings, new_names)
                fi.save(faiss_path)
                
            except Exception as e:
                logging.warning(f"FAISS update failed for {student_id}: {e}")
            
            messagebox.showinfo("Success", f"Student {name} ({student_id}) registered successfully")
            self._cancel()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to register student: {str(e)}")
    
    def _cancel(self):
        """Cancel registration and close dialog"""
        self.camera_running = False
        if not self.using_shared_camera:
            self.camera_manager.release()
            print("Released camera manager for registration")
        else:
            print("Keeping shared camera manager active")
        self.dialog.destroy()
    
    def _center_dialog(self):
        """Center the dialog on the parent window"""
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
