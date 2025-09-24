"""
Student Registration Dialog for BlazeFace-FRS System
Handles student registration with face capture and encoding
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
from typing import Optional, Dict, Any
import threading
import time

from ..core.standard_face_embedder import StandardFaceEmbedder
from ..core.blazeface_detector import BlazeFaceDetector
from ..core.database import DatabaseManager
from ..utils.camera_utils import CameraManager
from ..utils.encoding_quality_checker import EncodingQualityChecker

class StudentRegistrationDialog:
    def __init__(self, parent, database_manager: DatabaseManager, config: Dict[str, Any], 
                 existing_camera_manager: CameraManager = None):
        """
        Initialize student registration dialog
        
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
        self.face_embedder = StandardFaceEmbedder(model='large')
        
        # Use existing camera manager or create new one
        if existing_camera_manager and existing_camera_manager.is_camera_available():
            self.camera_manager = existing_camera_manager
            self.using_shared_camera = True
            print("Using shared camera manager")
        else:
            self.camera_manager = CameraManager(config.get('camera_index', 0))
            self.using_shared_camera = False
            print("Created new camera manager for registration")
            
        # Store reference for potential fallback
        self.fallback_camera_manager = None
            
        self.quality_checker = EncodingQualityChecker()
        
        # UI state
        self.is_capturing = False
        self.captured_faces = []
        self.current_frame = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Student Registration")
        self.dialog.geometry("800x600")
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
        
        # Start camera preview
        self._start_camera_preview()
        
        # Center dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Setup the registration dialog UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Form
        self._setup_form_panel(main_frame)
        
        # Right panel - Camera and Face Capture
        self._setup_camera_panel(main_frame)
        
        # Bottom panel - Buttons
        self._setup_button_panel(main_frame)
    
    def _setup_form_panel(self, parent):
        """Setup student information form"""
        form_frame = ttk.LabelFrame(parent, text="Student Information", padding="10")
        form_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Student ID
        ttk.Label(form_frame, text="Student ID *:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.student_id_var = tk.StringVar()
        self.student_id_entry = ttk.Entry(form_frame, textvariable=self.student_id_var, width=20)
        self.student_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Name
        ttk.Label(form_frame, text="Full Name *:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(form_frame, textvariable=self.name_var, width=20)
        self.name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Email
        ttk.Label(form_frame, text="Email:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.email_var = tk.StringVar()
        self.email_entry = ttk.Entry(form_frame, textvariable=self.email_var, width=20)
        self.email_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Phone
        ttk.Label(form_frame, text="Phone:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.phone_var = tk.StringVar()
        self.phone_entry = ttk.Entry(form_frame, textvariable=self.phone_var, width=20)
        self.phone_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Department
        ttk.Label(form_frame, text="Department:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.department_var = tk.StringVar()
        self.department_combo = ttk.Combobox(form_frame, textvariable=self.department_var, 
                                           values=["Computer Science", "Engineering", "Business", "Arts", "Science"])
        self.department_combo.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Year
        ttk.Label(form_frame, text="Year:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.year_var = tk.StringVar()
        self.year_combo = ttk.Combobox(form_frame, textvariable=self.year_var,
                                     values=["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduate"])
        self.year_combo.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Configure column weights
        form_frame.columnconfigure(1, weight=1)
    
    def _setup_camera_panel(self, parent):
        """Setup camera and face capture panel"""
        camera_frame = ttk.LabelFrame(parent, text="Face Capture", padding="10")
        camera_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera display - Use a Canvas for better image display
        self.camera_canvas = tk.Canvas(camera_frame, width=400, height=300, 
                                       background='black', highlightthickness=1,
                                       relief='sunken', bd=2)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Add initial text overlay for status
        self.camera_canvas.create_text(200, 150, text="Initializing camera...", 
                                       fill='white', font=('Arial', 12), tags='status_text')
        
        # Force canvas to update and be visible
        self.camera_canvas.update_idletasks()
        
        # Status label
        self.status_label = ttk.Label(camera_frame, text="Status: Initializing...", 
                                     font=('Arial', 9), foreground='blue')
        self.status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Face capture controls
        capture_frame = ttk.Frame(camera_frame)
        capture_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.capture_btn = ttk.Button(capture_frame, text="Capture Face", 
                                     command=self._capture_face, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_btn = ttk.Button(capture_frame, text="Clear Faces", 
                                   command=self._clear_faces)
        self.clear_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        self.debug_btn = ttk.Button(capture_frame, text="Debug Camera", 
                                   command=self._debug_camera)
        self.debug_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Captured faces display
        faces_frame = ttk.LabelFrame(camera_frame, text="Captured Faces", padding="5")
        faces_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame for captured faces
        canvas = tk.Canvas(faces_frame, height=150)
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
        
        # Instructions
        instructions = ttk.Label(camera_frame, 
                               text="Instructions:\n1. Position your face clearly in the camera view\n2. Ensure good lighting conditions\n3. Click 'Capture Face' when face is detected\n4. Capture multiple angles for better recognition\n5. Click 'Register Student' when done\n\nIf camera doesn't show, click 'Debug Camera' for troubleshooting",
                               justify=tk.LEFT)
        instructions.pack(fill=tk.X, pady=(10, 0))
    
    def _setup_button_panel(self, parent):
        """Setup button panel"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(button_frame, text="Register Student", 
                  command=self._register_student).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", 
                  command=self._cancel).pack(side=tk.RIGHT)
    
    def _start_camera_preview(self):
        """Start camera preview"""
        try:
            print(f"Starting camera preview - Using shared camera: {self.using_shared_camera}")
            print(f"Camera manager available: {self.camera_manager.is_camera_available()}")
            
            if self.camera_manager.is_camera_available():
                self.capture_btn.config(state=tk.NORMAL)
                self.camera_canvas.delete('all')
                self.camera_canvas.create_text(200, 150, text="Initializing camera...", 
                                              fill='white', font=('Arial', 12), tags='status_text')
                self.status_label.config(text="Status: Initializing camera...", foreground='blue')
                
                # Ensure canvas is properly sized and visible
                self._ensure_canvas_visible()
                
                # Add small delay to ensure camera is ready
                self.dialog.after(500, self._update_camera_preview)
            else:
                # Try fallback camera if shared camera doesn't work
                if self.using_shared_camera and not self.fallback_camera_manager:
                    print("Shared camera not available, trying fallback...")
                    self.fallback_camera_manager = CameraManager(self.config.get('camera_index', 0))
                    if self.fallback_camera_manager.is_camera_available():
                        self.camera_manager = self.fallback_camera_manager
                        self.using_shared_camera = False
                        print("Fallback camera initialized successfully")
                        self.capture_btn.config(state=tk.NORMAL)
                        self.camera_canvas.delete('all')
                        self.camera_canvas.create_text(200, 150, text="Initializing fallback camera...", 
                                                      fill='orange', font=('Arial', 12), tags='status_text')
                        self.status_label.config(text="Status: Using fallback camera...", foreground='orange')
                        self.dialog.after(500, self._update_camera_preview)
                        return
                
                self.camera_canvas.delete('all')
                self.camera_canvas.create_text(200, 150, text="Camera not available", 
                                              fill='red', font=('Arial', 12), tags='status_text')
                self.status_label.config(text="Status: Camera not available", foreground='red')
                print("Camera not available in student registration")
        except Exception as e:
            print(f"Error starting camera preview: {e}")
            self.camera_canvas.delete('all')
            self.camera_canvas.create_text(200, 150, text=f"Camera error: {str(e)}", 
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
                self.camera_canvas.config(width=400, height=300)
                self.camera_canvas.update_idletasks()
                # Canvas resized - removed verbose logging
            
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
        """Update camera preview"""
        try:
            if self.camera_manager.is_camera_available():
                ret, frame = self.camera_manager.get_frame()
                if ret and frame is not None:
                    # Camera frame received - removed verbose logging
                    
                    # Detect faces in frame
                    faces = self.face_detector.detect_faces(frame)
                    
                    # Draw face detection boxes
                    if faces:
                        frame = self.face_detector.draw_faces(frame, faces)
                    
                    # Resize frame to fit display
                    display_frame = cv2.resize(frame, (400, 300))
                    
                    # Convert frame for display using Canvas
                    try:
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        
                        # Clear canvas and display image
                        self.camera_canvas.delete('all')
                        self.camera_canvas.create_image(200, 150, image=frame_tk, anchor=tk.CENTER)
                        self.camera_canvas.image = frame_tk  # Keep reference to prevent garbage collection
                        
                        # Force canvas update and ensure visibility
                        self.camera_canvas.update_idletasks()
                        
                        # Canvas image updated successfully - removed verbose logging
                        
                    except Exception as img_error:
                        print(f"Error converting/displaying image: {img_error}")
                        # Fallback: show text on canvas
                        self.camera_canvas.delete('all')
                        self.camera_canvas.create_text(200, 150, text=f"Camera Active\nFrame: {frame.shape}", 
                                                      fill='green', font=('Arial', 12), tags='status_text')
                        self.camera_canvas.update_idletasks()
                    
                    # Update status
                    face_count = len(faces)
                    self.status_label.config(text=f"Status: Camera active - {face_count} face(s) detected", 
                                           foreground='green')
                    
                    # Store original frame for face capture
                    self.current_frame = frame
                else:
                    print("No camera frame received")
                    self.camera_canvas.delete('all')
                    self.camera_canvas.create_text(200, 150, text="No camera feed\nClick 'Capture Face' to retry", 
                                                  fill='red', font=('Arial', 12), tags='status_text')
                    self.status_label.config(text="Status: No camera feed", foreground='red')
            else:
                print("Camera not available during update")
                self.camera_canvas.delete('all')
                self.camera_canvas.create_text(200, 150, text="Camera not available", 
                                              fill='red', font=('Arial', 12), tags='status_text')
                self.status_label.config(text="Status: Camera not available", foreground='red')
            
            # Schedule next update
            self.dialog.after(50, self._update_camera_preview)
            
        except Exception as e:
            print(f"Error updating camera preview: {e}")
            self.camera_canvas.delete('all')
            self.camera_canvas.create_text(200, 150, text=f"Camera error: {str(e)}", 
                                          fill='red', font=('Arial', 12), tags='status_text')
            self.status_label.config(text=f"Status: Error - {str(e)}", foreground='red')
            self.dialog.after(1000, self._update_camera_preview)
    
    def _capture_face(self):
        """Capture face from current frame"""
        try:
            if self.current_frame is None:
                messagebox.showwarning("Warning", "No camera feed available. Please wait for camera to initialize.")
                return
            
            # Detect faces
            faces = self.face_detector.detect_faces(self.current_frame)
            
            if not faces:
                messagebox.showwarning("Warning", "No face detected in current frame.\n\nPlease ensure:\n- Your face is clearly visible\n- Good lighting conditions\n- Face is centered in the camera view")
                return
            
            # Use the first detected face
            face_box = faces[0]
            x, y, w, h, confidence = face_box
            
            # Check confidence threshold
            if confidence < 0.5:
                messagebox.showwarning("Warning", f"Face detection confidence too low: {confidence:.2f}\n\nPlease ensure better lighting and face positioning.")
                return
            
            # Extract face region
            face_region = self.face_detector.extract_face_region(self.current_frame, (x, y, w, h))
            
            if face_region is not None and face_region.size > 0:
                # Get face embedding
                embedding = self.face_embedder.get_embedding(face_region)
                
                if embedding is not None:
                    # Check encoding quality before storing
                    student_id = self.student_id_var.get().strip()
                    
                    # Get existing encodings for this student
                    existing_encodings = []
                    if student_id:
                        existing_encs = self.database_manager.get_face_encodings(student_id)
                        existing_encodings = [enc for _, enc, _ in existing_encs]
                    
                    # Get encodings for other students
                    other_encodings = []
                    all_encs = self.database_manager.get_face_encodings()
                    for sid, enc, _ in all_encs:
                        if sid != student_id:
                            other_encodings.append(enc)
                    
                    # Check quality
                    is_acceptable, reason, quality_metrics = self.quality_checker.check_new_encoding_quality(
                        embedding, existing_encodings, other_encodings, student_id
                    )
                    
                    if not is_acceptable:
                        messagebox.showwarning("Quality Check Failed", 
                            f"Face encoding quality check failed:\n\n{reason}\n\n"
                            f"Please try again with:\n"
                            f"- Better lighting\n"
                            f"- Clearer face positioning\n"
                            f"- Different angle")
                        return
                    
                    # Store captured face (image will be saved during registration)
                    face_data = {
                        'image': face_region,
                        'embedding': embedding,
                        'confidence': confidence,
                        'timestamp': time.time(),
                        'quality_metrics': quality_metrics,
                        'image_path': None  # Will be set during registration
                    }
                    self.captured_faces.append(face_data)
                    
                    # Update display
                    self._update_captured_faces_display()
                    
                    quality_score = quality_metrics.get('quality_score', 0.0)
                    messagebox.showinfo("Success", 
                        f"Face captured successfully!\n"
                        f"Confidence: {confidence:.2f}\n"
                        f"Quality Score: {quality_score:.2f}\n\n"
                        f"You can capture multiple angles for better recognition.")
                else:
                    messagebox.showerror("Error", "Failed to generate face embedding. Please try again.")
            else:
                messagebox.showerror("Error", "Failed to extract face region. Please try again.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture face: {str(e)}")
            print(f"Face capture error: {e}")
    
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
                face_image_pil.thumbnail((50, 50), Image.Resampling.LANCZOS)
                face_image_tk = ImageTk.PhotoImage(face_image_pil)
                
                face_label = ttk.Label(face_frame, image=face_image_tk)
                face_label.image = face_image_tk
                face_label.pack(side=tk.LEFT, padx=(0, 10))
                
                # Face info
                info_text = f"Face {i+1}\nConfidence: {face_data['confidence']:.2f}"
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
            debug_text += f"Current Frame: {self.current_frame is not None}\n"
            if self.current_frame is not None:
                debug_text += f"Frame Shape: {self.current_frame.shape}\n"
            debug_text += f"Camera Info: {info}\n"
            
            # Canvas debug info
            debug_text += f"\nCanvas Debug:\n"
            debug_text += f"Canvas Size: {self.camera_canvas.winfo_width()}x{self.camera_canvas.winfo_height()}\n"
            debug_text += f"Canvas Visible: {self.camera_canvas.winfo_viewable()}\n"
            debug_text += f"Canvas Items: {len(self.camera_canvas.find_all())}\n"
            
            # Test face detection
            if self.current_frame is not None:
                faces = self.face_detector.detect_faces(self.current_frame)
                debug_text += f"Faces Detected: {len(faces)}\n"
                if faces:
                    debug_text += f"First Face: {faces[0]}\n"
            
            # Test canvas image display
            debug_text += f"\nTesting Canvas Display...\n"
            try:
                # Create a test rectangle to verify canvas is working
                self.camera_canvas.delete('all')
                self.camera_canvas.create_rectangle(50, 50, 350, 250, fill='red', outline='white', width=2)
                self.camera_canvas.create_text(200, 150, text="Canvas Test", fill='white', font=('Arial', 16))
                self.camera_canvas.update_idletasks()
                debug_text += "Canvas test rectangle created successfully\n"
            except Exception as e:
                debug_text += f"Canvas test failed: {str(e)}\n"
            
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
            
            # Save face images and add face encodings with multi-image processing
            success_count = 0
            face_data_dir = self.config.get('face_data_path', 'face_data')
            os.makedirs(face_data_dir, exist_ok=True)
            
            # Determine embedder type for encoding storage
            embedder_type = 'standard_face'  # Default to new embedder
            if hasattr(self.face_embedder, '__class__'):
                if 'InsightFace' in self.face_embedder.__class__.__name__:
                    embedder_type = 'insightface'
                elif 'Standard' in self.face_embedder.__class__.__name__:
                    embedder_type = 'standard_face'
                elif 'FaceNet' in self.face_embedder.__class__.__name__:
                    embedder_type = 'facenet'
                elif 'Simple' in self.face_embedder.__class__.__name__:
                    embedder_type = 'simple'
            
            # Process multiple face captures with quality assessment
            processed_embeddings = []
            valid_images = []
            
            for i, face_data in enumerate(self.captured_faces):
                # Generate unique filename for this face
                timestamp = int(time.time() * 1000) + i  # milliseconds + index for uniqueness
                face_filename = f"{student_data['student_id']}_{timestamp}.jpg"
                face_filepath = os.path.join(face_data_dir, face_filename)
                
                # Save face image
                cv2.imwrite(face_filepath, face_data['image'])
                
                # Validate embedding quality
                embedding = face_data['embedding']
                if embedding is not None and len(embedding) > 0:
                    # Check embedding quality (not all zeros, reasonable norm)
                    embedding_norm = np.linalg.norm(embedding)
                    if embedding_norm > 0.1:  # Reasonable threshold for valid embedding
                        processed_embeddings.append(embedding)
                        valid_images.append(face_filepath)
                        
                        # Add individual face encoding with image path
                        success = self.database_manager.add_face_encoding(
                            student_data['student_id'],
                            embedding,
                            embedder_type,
                            face_filepath
                        )
                        if success:
                            success_count += 1
                    else:
                        print(f"Warning: Low quality embedding for face {i}, skipping")
                else:
                    print(f"Warning: Invalid embedding for face {i}, skipping")
            
            # Create averaged embedding for better recognition (if multiple valid embeddings)
            if len(processed_embeddings) > 1:
                try:
                    # Average the embeddings for better recognition
                    avg_embedding = np.mean(processed_embeddings, axis=0)
                    avg_filename = f"{student_data['student_id']}_averaged.jpg"
                    avg_filepath = os.path.join(face_data_dir, avg_filename)
                    
                    # Save a representative image (first valid image)
                    if valid_images:
                        cv2.imwrite(avg_filepath, cv2.imread(valid_images[0]))
                    
                    # Store averaged embedding
                    success = self.database_manager.add_face_encoding(
                        student_data['student_id'],
                        avg_embedding,
                        f"{embedder_type}_averaged",
                        avg_filepath
                    )
                    if success:
                        success_count += 1
                        print(f"Added averaged embedding from {len(processed_embeddings)} captures")
                        
                except Exception as e:
                    print(f"Error creating averaged embedding: {e}")
            
            # Log registration statistics
            print(f"Registration Summary:")
            print(f"  Total captures: {len(self.captured_faces)}")
            print(f"  Valid embeddings: {len(processed_embeddings)}")
            print(f"  Stored encodings: {success_count}")
            print(f"  Embedder type: {embedder_type}")
            
            if success_count > 0:
                messagebox.showinfo("Success", 
                                  f"Student registered successfully!\n"
                                  f"Added {success_count} face encodings")
                self._cancel()
            else:
                messagebox.showerror("Error", "Failed to add face encodings")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to register student: {str(e)}")
    
    def _cancel(self):
        """Cancel registration and close dialog"""
        try:
            # Release fallback camera if we created it
            if self.fallback_camera_manager:
                self.fallback_camera_manager.release()
                print("Released fallback camera manager")
            
            # Only release main camera if we created our own (not shared)
            if not self.using_shared_camera and not self.fallback_camera_manager:
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
