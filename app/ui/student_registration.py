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

from ..core.simple_face_embedder import SimpleFaceEmbedder
from ..core.blazeface_detector import BlazeFaceDetector
from ..core.database import DatabaseManager
from ..utils.camera_utils import CameraManager

class StudentRegistrationDialog:
    def __init__(self, parent, database_manager: DatabaseManager, config: Dict[str, Any]):
        """
        Initialize student registration dialog
        
        Args:
            parent: Parent window
            database_manager: Database manager instance
            config: Application configuration
        """
        self.parent = parent
        self.database_manager = database_manager
        self.config = config
        
        # Initialize components
        self.face_detector = BlazeFaceDetector(
            min_detection_confidence=config.get('detection_confidence', 0.7)
        )
        self.face_embedder = SimpleFaceEmbedder()
        self.camera_manager = CameraManager(config.get('camera_index', 0))
        
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
        
        # Camera display
        self.camera_label = ttk.Label(camera_frame, text="Initializing camera...", 
                                     anchor=tk.CENTER, background='black', foreground='white')
        self.camera_label.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
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
            if self.camera_manager.is_camera_available():
                self.capture_btn.config(state=tk.NORMAL)
                self.camera_label.config(text="Initializing camera...")
                self.status_label.config(text="Status: Initializing camera...", foreground='blue')
                self._update_camera_preview()
            else:
                self.camera_label.config(text="Camera not available")
                self.status_label.config(text="Status: Camera not available", foreground='red')
        except Exception as e:
            print(f"Error starting camera preview: {e}")
            self.camera_label.config(text=f"Camera error: {str(e)}")
            self.status_label.config(text=f"Status: Error - {str(e)}", foreground='red')
    
    def _update_camera_preview(self):
        """Update camera preview"""
        try:
            if self.camera_manager.is_camera_available():
                ret, frame = self.camera_manager.get_frame()
                if ret and frame is not None:
                    # Detect faces in frame
                    faces = self.face_detector.detect_faces(frame)
                    
                    # Draw face detection boxes
                    if faces:
                        frame = self.face_detector.draw_faces(frame, faces)
                    
                    # Resize frame to fit display
                    display_frame = cv2.resize(frame, (400, 300))
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update display
                    self.camera_label.config(image=frame_tk, text="")
                    self.camera_label.image = frame_tk
                    
                    # Update status
                    face_count = len(faces)
                    self.status_label.config(text=f"Status: Camera active - {face_count} face(s) detected", 
                                           foreground='green')
                    
                    # Store original frame for face capture
                    self.current_frame = frame
                else:
                    self.camera_label.config(text="No camera feed - Click 'Capture Face' to retry")
                    self.status_label.config(text="Status: No camera feed", foreground='red')
            
            # Schedule next update
            self.dialog.after(50, self._update_camera_preview)
            
        except Exception as e:
            print(f"Error updating camera preview: {e}")
            self.camera_label.config(text=f"Camera error: {str(e)}")
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
                    # Store captured face
                    face_data = {
                        'image': face_region,
                        'embedding': embedding,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    self.captured_faces.append(face_data)
                    
                    # Update display
                    self._update_captured_faces_display()
                    
                    messagebox.showinfo("Success", f"Face captured successfully!\nConfidence: {confidence:.2f}\n\nYou can capture multiple angles for better recognition.")
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
            
            # Test face detection
            if self.current_frame is not None:
                faces = self.face_detector.detect_faces(self.current_frame)
                debug_text += f"Faces Detected: {len(faces)}\n"
                if faces:
                    debug_text += f"First Face: {faces[0]}\n"
            
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
            
            # Add face encodings
            success_count = 0
            for face_data in self.captured_faces:
                success = self.database_manager.add_face_encoding(
                    student_data['student_id'],
                    face_data['embedding']
                )
                if success:
                    success_count += 1
            
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
            self.camera_manager.release()
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
