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
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                # Draw face detection
                display_frame = frame.copy()
                face_detected = False
                
                for face in faces:
                    x, y, w, h, confidence = face
                    if confidence > 0.7:  # Good confidence threshold
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Face: {confidence:.2f}", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        face_detected = True
                
                # Convert to PhotoImage and update display
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(display_frame, (500, 400))
                
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image)
                
                # Update UI in main thread
                self.dialog.after(0, self._update_camera_display, photo, face_detected)
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Camera loop error: {e}")
            self.dialog.after(0, lambda: self.status_label.config(text="Camera error", foreground='red'))
    
    def _update_camera_display(self, photo, face_detected):
        """Update camera display in main thread"""
        try:
            # Check if dialog still exists
            if not self.dialog.winfo_exists():
                return
                
            self.display_canvas.delete('all')
            self.display_canvas.create_image(250, 200, image=photo, anchor=tk.CENTER)
            self.display_canvas.image = photo  # Keep a reference
            
            # Update status
            if face_detected:
                self.status_label.config(text="Status: Face detected - Ready to capture", foreground='green')
            else:
                self.status_label.config(text="Status: Position your face in the camera", foreground='blue')
                
        except tk.TclError:
            # Dialog was closed, stop camera loop
            return
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
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
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if not faces:
                messagebox.showwarning("Warning", "No face detected in current frame")
                return
            
            # Use the best face (highest confidence)
            best_face = max(faces, key=lambda f: f[4])
            x, y, w, h, confidence = best_face
            
            if confidence < 0.7:
                messagebox.showwarning("Warning", f"Face detection confidence too low: {confidence:.2f}")
                return
            
            # Extract face region
            face_region = self.face_detector.extract_face_region(frame, (x, y, w, h))
            
            if face_region is not None and face_region.size > 0:
                # Get face embedding
                embedder = self.embedders[self.current_embedder]
                embedding = embedder.get_embedding(face_region)
                
                if embedding is not None:
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
                    
                    quality_threshold = self.quality_var.get()
                    if quality_metrics.get('quality_score', 0) < quality_threshold:
                        is_acceptable = False
                        reason = f"Quality score {quality_metrics.get('quality_score', 0):.2f} below threshold {quality_threshold:.2f}"
                    
                    if not is_acceptable:
                        messagebox.showwarning("Quality Check Failed", 
                            f"Face encoding quality check failed:\n\n{reason}\n\n"
                            f"Please try again with:\n"
                            f"- Better lighting\n"
                            f"- Clearer face positioning\n"
                            f"- Different angle")
                        return
                    
                    # Store captured face
                    face_data = {
                        'image': face_region,
                        'embedding': embedding,
                        'confidence': confidence,
                        'timestamp': time.time(),
                        'quality_metrics': quality_metrics,
                        'source': 'camera',
                        'image_path': None
                    }
                    self.captured_faces.append(face_data)
                    
                    # Update display
                    self._update_captured_faces_display()
                    
                    quality_score = quality_metrics.get('quality_score', 0.0)
                    messagebox.showinfo("Success", 
                        f"Face captured successfully!\n"
                        f"Confidence: {confidence:.2f}\n"
                        f"Quality Score: {quality_score:.2f}\n"
                        f"Encoder: {self.current_embedder}")
                else:
                    messagebox.showerror("Error", "Failed to generate face embedding")
            else:
                messagebox.showerror("Error", "Failed to extract face region")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture face: {str(e)}")
    
    def _select_images(self):
        """Select high-quality images for face processing"""
        try:
            filetypes = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
            
            filenames = filedialog.askopenfilenames(
                title="Select High-Quality Student Photos",
                filetypes=filetypes
            )
            
            if filenames:
                self.uploaded_images = list(filenames)
                self._update_image_list()
                self.process_btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"Status: {len(filenames)} images selected", foreground='green')
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select images: {str(e)}")
    
    def _update_image_list(self):
        """Update the image listbox"""
        try:
            self.image_listbox.delete(0, tk.END)
            for i, image_path in enumerate(self.uploaded_images):
                filename = os.path.basename(image_path)
                self.image_listbox.insert(tk.END, f"{i+1}. {filename}")
        except Exception as e:
            print(f"Error updating image list: {e}")
    
    def _preview_selected_image(self, event=None):
        """Preview selected image"""
        try:
            selection = self.image_listbox.curselection()
            if not selection:
                return
            
            index = selection[0]
            if 0 <= index < len(self.uploaded_images):
                image_path = self.uploaded_images[index]
                
                # Load and display image
                image = cv2.imread(image_path)
                if image is not None:
                    # Resize to fit display
                    height, width = image.shape[:2]
                    scale = min(500 / width, 400 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    resized = cv2.resize(image, (new_width, new_height))
                    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PhotoImage
                    pil_image = Image.fromarray(rgb_image)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update display
                    self.display_canvas.delete('all')
                    self.display_canvas.create_image(250, 200, image=photo, anchor=tk.CENTER)
                    self.display_canvas.image = photo
                    
                    self.status_label.config(text=f"Preview: {os.path.basename(image_path)}", foreground='blue')
                    
        except Exception as e:
            print(f"Error previewing image: {e}")
    
    def _process_uploaded_images(self):
        """Process uploaded images to extract faces"""
        try:
            if not self.uploaded_images:
                messagebox.showwarning("Warning", "No images selected")
                return
            
            embedder = self.embedders[self.current_embedder]
            processed_count = 0
            failed_count = 0
            
            for image_path in self.uploaded_images:
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        failed_count += 1
                        continue
                    
                    # Detect faces
                    faces = self.face_detector.detect_faces(image)
                    
                    if not faces:
                        failed_count += 1
                        continue
                    
                    # Use the best face (largest area)
                    best_face = max(faces, key=lambda f: f[2] * f[3])  # w * h
                    x, y, w, h, confidence = best_face
                    
                    # Extract face region
                    face_region = self.face_detector.extract_face_region(image, (x, y, w, h))
                    
                    if face_region is not None and face_region.size > 0:
                        # Get face embedding
                        embedding = embedder.get_embedding(face_region)
                        
                        if embedding is not None:
                            # Check quality
                            quality_threshold = self.quality_var.get()
                            
                            # Simple quality check for images
                            quality_score = self._calculate_image_quality(face_region)
                            
                            if quality_score >= quality_threshold:
                                # Store captured face
                                face_data = {
                                    'image': face_region,
                                    'embedding': embedding,
                                    'confidence': confidence,
                                    'timestamp': time.time(),
                                    'quality_metrics': {'quality_score': quality_score},
                                    'source': 'image',
                                    'image_path': image_path
                                }
                                self.captured_faces.append(face_data)
                                processed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    failed_count += 1
            
            # Update display
            self._update_captured_faces_display()
            
            messagebox.showinfo("Processing Complete", 
                              f"Processed {len(self.uploaded_images)} images\n"
                              f"Successfully extracted: {processed_count} faces\n"
                              f"Failed: {failed_count} images\n"
                              f"Quality threshold: {self.quality_var.get():.2f}")
            
            self.status_label.config(text=f"Status: Processed {processed_count} faces from images", foreground='green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process images: {str(e)}")
    
    def _calculate_image_quality(self, face_image):
        """Calculate quality score for face image"""
        try:
            height, width = face_image.shape[:2]
            
            # Size check
            if width < 100 or height < 100:
                return 0.0
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Brightness check
            brightness = np.mean(gray)
            if brightness < 50 or brightness > 200:
                return 0.3
            
            # Contrast check
            contrast = np.std(gray)
            if contrast < 20:
                return 0.4
            
            # Blur check
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:
                return 0.5
            
            # Calculate overall quality score
            size_score = min(1.0, (width * height) / (200 * 200))
            brightness_score = 1.0 - abs(brightness - 128) / 128
            contrast_score = min(1.0, contrast / 50)
            blur_score_normalized = min(1.0, blur_score / 500)
            
            overall_score = (size_score + brightness_score + contrast_score + blur_score_normalized) / 4
            
            return overall_score
            
        except Exception as e:
            print(f"Error calculating image quality: {e}")
            return 0.0
    
    def _update_captured_faces_display(self):
        """Update the captured faces display"""
        try:
            # Clear existing display
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            # Add captured faces
            for i, face_data in enumerate(self.captured_faces):
                face_frame = ttk.Frame(self.scrollable_frame)
                face_frame.pack(fill=tk.X, pady=2)
                
                # Face thumbnail
                face_image = face_data['image']
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_image_pil = Image.fromarray(face_image_rgb)
                face_image_pil.thumbnail((60, 60), Image.Resampling.LANCZOS)
                face_image_tk = ImageTk.PhotoImage(face_image_pil)
                
                face_label = ttk.Label(face_frame, image=face_image_tk)
                face_label.image = face_image_tk
                face_label.pack(side=tk.LEFT, padx=(0, 10))
                
                # Face info
                source = face_data['source']
                quality_score = face_data['quality_metrics'].get('quality_score', 0.0)
                info_text = f"Face {i+1} ({source})\nConfidence: {face_data['confidence']:.2f}\nQuality: {quality_score:.2f}"
                info_label = ttk.Label(face_frame, text=info_text, justify=tk.LEFT)
                info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                # Remove button
                remove_btn = ttk.Button(face_frame, text="Remove", 
                                      command=lambda idx=i: self._remove_face(idx))
                remove_btn.pack(side=tk.RIGHT)
                
        except Exception as e:
            print(f"Error updating captured faces display: {e}")
    
    def _remove_face(self, index: int):
        """Remove a captured face"""
        try:
            if 0 <= index < len(self.captured_faces):
                del self.captured_faces[index]
                self._update_captured_faces_display()
        except Exception as e:
            print(f"Error removing face: {e}")
    
    def _clear_faces(self):
        """Clear all captured faces"""
        try:
            self.captured_faces.clear()
            self._update_captured_faces_display()
        except Exception as e:
            print(f"Error clearing faces: {e}")
    
    def _debug_camera(self):
        """Debug camera functionality"""
        try:
            info = self.camera_manager.get_camera_info()
            debug_text = f"Camera Debug Info:\n"
            debug_text += f"Available: {self.camera_manager.is_camera_available()}\n"
            debug_text += f"Camera Info: {info}\n"
            debug_text += f"Registration Mode: {self.registration_mode}\n"
            debug_text += f"Current Embedder: {self.current_embedder}\n"
            debug_text += f"Captured Faces: {len(self.captured_faces)}\n"
            
            messagebox.showinfo("Camera Debug", debug_text)
            
        except Exception as e:
            messagebox.showerror("Debug Error", f"Debug failed: {str(e)}")
    
    def _register_student(self):
        """Register the student with captured faces"""
        try:
            # Validate form data
            if not self.student_id_var.get().strip():
                messagebox.showerror("Error", "Student ID is required")
                return
            
            if not self.name_var.get().strip():
                messagebox.showerror("Error", "Name is required")
                return
            
            if not self.captured_faces:
                messagebox.showerror("Error", "At least one face capture is required")
                return
            
            # Check if student already exists
            existing_student = self.database_manager.get_student(self.student_id_var.get().strip())
            if existing_student:
                if not messagebox.askyesno("Student Exists", 
                                         f"Student ID {self.student_id_var.get()} already exists. Update existing record?"):
                    return
            
            # Prepare student data
            student_data = {
                'student_id': self.student_id_var.get().strip(),
                'name': self.name_var.get().strip(),
                'email': self.email_var.get().strip() or None,
                'phone': self.phone_var.get().strip() or None,
                'department': self.department_var.get().strip() or None,
                'year': self.year_var.get().strip() or None
            }
            
            # Add/Update student
            if existing_student:
                success = self.database_manager.update_student(student_data['student_id'], student_data)
                if not success:
                    messagebox.showerror("Error", "Failed to update student record")
                    return
            else:
                success = self.database_manager.add_student(student_data)
                if not success:
                    messagebox.showerror("Error", "Failed to add student record")
                    return
            
            # Save face images and add face encodings
            success_count = 0
            face_data_dir = self.config.get('face_data_path', 'face_data')
            os.makedirs(face_data_dir, exist_ok=True)
            
            embedder_type = self.current_embedder
            
            # Process face captures
            for i, face_data in enumerate(self.captured_faces):
                # Generate unique filename
                timestamp = int(time.time() * 1000) + i
                face_filename = f"{student_data['student_id']}_{timestamp}.jpg"
                face_filepath = os.path.join(face_data_dir, face_filename)
                
                # Save face image
                cv2.imwrite(face_filepath, face_data['image'])
                
                # Add face encoding
                success = self.database_manager.add_face_encoding(
                    student_data['student_id'],
                    face_data['embedding'],
                    embedder_type,
                    face_filepath
                )
                if success:
                    success_count += 1
                    # Add to FAISS index if available
                    try:
                        fi = FaissIndex()
                        base = os.path.join('data', 'faiss')
                        if os.path.exists(base + '.pkl') or os.path.exists(base + '.index'):
                            fi.load(base)
                        # add this embedding
                        emb = face_data['embedding'].astype('float32')
                        fi.add(emb.reshape(1, -1), [student_data['student_id']])
                        fi.save(base)
                    except Exception as e:
                        # Non-fatal: log and continue
                        logging.getLogger(__name__).warning(f"FAISS update failed for {student_data['student_id']}: {e}")
            
            if success_count > 0:
                messagebox.showinfo("Success", 
                                  f"Student registered successfully!\n"
                                  f"Added {success_count} face encodings\n"
                                  f"Encoder: {embedder_type}\n"
                                  f"Mode: {self.registration_mode}")
                self._cancel()
            else:
                messagebox.showerror("Error", "Failed to add face encodings")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to register student: {str(e)}")
    
    def _cancel(self):
        """Cancel registration and close dialog"""
        try:
            self.camera_running = False
            
            # Only release camera if we created our own (not shared)
            if not self.using_shared_camera:
                self.camera_manager.release()
                print("Released camera manager for registration")
            else:
                print("Keeping shared camera manager active")
                
            self.dialog.destroy()
        except Exception as e:
            print(f"Error closing dialog: {e}")
    
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
