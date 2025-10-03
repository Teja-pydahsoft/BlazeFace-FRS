"""
Mobile-Style Student Registration Dialog
No photo storage, pickle-based encodings with progress indicator
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import pickle
import threading
import time
from typing import Optional, Dict, Any, List

from ..core.simple_face_embedder import SimpleFaceEmbedder
from ..core.blazeface_detector import BlazeFaceDetector
from ..core.database import DatabaseManager
from ..utils.camera_utils import CameraManager
from ..utils.encoding_quality_checker import EncodingQualityChecker

class MobileStyleRegistrationDialog:
    def __init__(self, parent, database_manager: DatabaseManager, config: Dict[str, Any], 
                 existing_camera_manager: CameraManager = None):
        """
        Initialize mobile-style registration dialog
        
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
        self.face_embedder = SimpleFaceEmbedder()
        self.quality_checker = EncodingQualityChecker()
        
        # Use existing camera manager or create new one
        if existing_camera_manager and existing_camera_manager.is_camera_available():
            self.camera_manager = existing_camera_manager
            self.using_shared_camera = True
            print("Using shared camera manager")
        else:
            self.camera_manager = CameraManager(config.get('camera_index', 0))
            self.using_shared_camera = False
            print("Created new camera manager for registration")
        
        # Registration state
        self.is_registering = False
        self.registration_progress = 0
        self.required_captures = 5  # Number of face captures needed
        self.captured_encodings = []
        self.registration_thread = None
        self.camera_running = True
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Face Registration")
        self.dialog.geometry("600x500")
        self.dialog.resizable(False, False)
        
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
        """Setup the mobile-style registration UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Registration", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Student ID input
        id_frame = ttk.Frame(main_frame)
        id_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(id_frame, text="Student ID:", font=('Arial', 12)).pack(side=tk.LEFT)
        self.student_id_var = tk.StringVar()
        self.student_id_entry = ttk.Entry(id_frame, textvariable=self.student_id_var, 
                                         font=('Arial', 12), width=20)
        self.student_id_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Camera preview
        camera_frame = ttk.LabelFrame(main_frame, text="Face Detection", padding="10")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.camera_label = ttk.Label(camera_frame, text="Initializing camera...", 
                                     font=('Arial', 12))
        self.camera_label.pack(expand=True)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Registration Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(pady=(0, 10))
        
        # Progress label
        self.progress_label = ttk.Label(progress_frame, text="Ready to start registration", 
                                       font=('Arial', 12))
        self.progress_label.pack()
        
        # Status label
        self.status_label = ttk.Label(progress_frame, text="Position your face in the camera", 
                                     font=('Arial', 10), foreground='blue')
        self.status_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(button_frame, text="Start Registration", 
                                      command=self._start_registration)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", 
                                       command=self._cancel)
        self.cancel_button.pack(side=tk.RIGHT)
        
        # Initially disable start button
        self.start_button.config(state='disabled')
    
    def _start_camera_preview(self):
        """Start camera preview"""
        try:
            if not self.camera_manager.is_camera_available():
                self.camera_label.config(text="Camera not available", foreground='red')
                return
            
            # Start camera in a separate thread
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.camera_label.config(text="Camera error", foreground='red')
    
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
                    if confidence > 0.8:  # Higher confidence threshold for better accuracy
                        # Additional face quality checks
                        if self._is_valid_face_region(x, y, w, h, frame):
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Face: {confidence:.2f}", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            face_detected = True
                        else:
                            # Show rejected faces in red
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(display_frame, f"Rejected: {confidence:.2f}", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Convert to PhotoImage and update label
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(display_frame, (400, 300))
                
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image)
                
                # Update UI in main thread
                self.dialog.after(0, self._update_camera_display, photo, face_detected)
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Camera loop error: {e}")
            self.dialog.after(0, lambda: self.camera_label.config(text="Camera error", foreground='red'))
    
    def _update_camera_display(self, photo, face_detected):
        """Update camera display in main thread"""
        try:
            # Check if dialog still exists
            if not self.dialog.winfo_exists():
                return
                
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep a reference
            
            # Enable start button if face is detected and student ID is entered
            if face_detected and self.student_id_var.get().strip():
                self.start_button.config(state='normal')
                self.status_label.config(text="Face detected! Click 'Start Registration'", 
                                        foreground='green')
            elif not face_detected:
                self.start_button.config(state='disabled')
                self.status_label.config(text="Position your face in the camera", 
                                        foreground='blue')
            elif not self.student_id_var.get().strip():
                self.start_button.config(state='disabled')
                self.status_label.config(text="Enter Student ID first", 
                                        foreground='orange')
                
        except tk.TclError:
            # Dialog was closed, stop camera loop
            return
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
    def _start_registration(self):
        """Start the face registration process"""
        student_id = self.student_id_var.get().strip()
        if not student_id:
            messagebox.showerror("Error", "Please enter a Student ID")
            return
        
        # Check if student already exists
        existing_student = self.database_manager.get_student(student_id)
        if existing_student:
            if not messagebox.askyesno("Student Exists", 
                                     f"Student ID {student_id} already exists. Update existing record?"):
                return
        
        # Start registration
        self.is_registering = True
        self.registration_progress = 0
        self.captured_encodings = []
        
        # Update UI
        self.start_button.config(state='disabled')
        self.student_id_entry.config(state='disabled')
        
        # Start registration thread
        self.registration_thread = threading.Thread(target=self._registration_loop, daemon=True)
        self.registration_thread.start()
    
    def _registration_loop(self):
        """Main registration loop"""
        try:
            capture_count = 0
            last_capture_time = 0
            capture_interval = 1.0  # 1 second between captures
            
            while capture_count < self.required_captures and self.is_registering:
                current_time = time.time()
                
                # Check if enough time has passed since last capture
                if current_time - last_capture_time < capture_interval:
                    time.sleep(0.1)
                    continue
                
                # Get current frame
                ret, frame = self.camera_manager.get_frame()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                if not faces:
                    self.dialog.after(0, lambda: self.status_label.config(
                        text="No face detected. Please position your face in the camera", 
                        foreground='orange'))
                    time.sleep(0.1)
                    continue
                
                # Get the best face (highest confidence)
                best_face = max(faces, key=lambda f: f[4])  # f[4] is confidence
                x, y, w, h, confidence = best_face
                
                if confidence < 0.8:  # Require very high confidence
                    self.dialog.after(0, lambda: self.status_label.config(
                        text="Face not clear enough. Please look directly at the camera", 
                        foreground='orange'))
                    time.sleep(0.1)
                    continue
                
                # Additional validation for registration
                if not self._is_valid_face_region(x, y, w, h, frame):
                    self.dialog.after(0, lambda: self.status_label.config(
                        text="Invalid face region detected. Please adjust position", 
                        foreground='orange'))
                    time.sleep(0.1)
                    continue
                
                # Extract face region
                face_region = self.face_detector.extract_face_region(frame, (x, y, w, h))
                if face_region is None:
                    time.sleep(0.1)
                    continue
                
                # Get face embedding
                embedding = self.face_embedder.get_embedding(face_region)
                if embedding is None:
                    time.sleep(0.1)
                    continue
                
                # Check quality
                student_id = self.student_id_var.get().strip()
                
                # Get existing encodings for this student
                existing_encodings = []
                if student_id:
                    existing_encs = self.database_manager.get_face_encodings(student_id)
                    existing_encodings = [enc for _, enc, _, _ in existing_encs]
                
                # Get encodings for other students
                other_encodings = []
                all_encs = self.database_manager.get_face_encodings()
                for sid, enc, _, _ in all_encs:
                    if sid != student_id:
                        other_encodings.append(enc)
                
                # Check quality
                is_acceptable, reason, quality_metrics = self.quality_checker.check_new_encoding_quality(
                    embedding, existing_encodings, other_encodings, student_id
                )
                
                if not is_acceptable:
                    self.dialog.after(0, lambda: self.status_label.config(
                        text=f"Quality check failed: {reason}", 
                        foreground='red'))
                    time.sleep(0.5)
                    continue
                
                # Acceptable encoding - add to collection
                self.captured_encodings.append(embedding)
                capture_count += 1
                last_capture_time = current_time
                
                # Update progress
                progress = (capture_count / self.required_captures) * 100
                self.dialog.after(0, self._update_progress, progress, capture_count)
                
                # Brief pause to show progress
                time.sleep(0.5)
            
            if self.is_registering:  # Registration completed successfully
                self.dialog.after(0, self._complete_registration)
                
        except Exception as e:
            print(f"Registration error: {e}")
            self.dialog.after(0, lambda: self.status_label.config(
                text=f"Registration error: {str(e)}", 
                foreground='red'))
            self.dialog.after(0, self._reset_registration)
    
    def _update_progress(self, progress, capture_count):
        """Update progress display"""
        self.progress_var.set(progress)
        self.progress_label.config(text=f"Progress: {capture_count}/{self.required_captures} captures ({progress:.0f}%)")
        self.status_label.config(text=f"Captured {capture_count} face encodings...", 
                                foreground='green')
    
    def _complete_registration(self):
        """Complete the registration process"""
        try:
            student_id = self.student_id_var.get().strip()
            
            # Create averaged embedding
            if len(self.captured_encodings) > 1:
                averaged_encoding = np.mean(self.captured_encodings, axis=0)
            else:
                averaged_encoding = self.captured_encodings[0]
            
            # Add/Update student
            student_data = {
                'student_id': student_id,
                'name': f"Student {student_id}",  # Default name
                'email': None,
                'phone': None,
                'department': None,
                'year': None
            }
            
            existing_student = self.database_manager.get_student(student_id)
            if existing_student:
                success = self.database_manager.update_student(student_id, student_data)
            else:
                success = self.database_manager.add_student(student_data)
            
            if not success:
                messagebox.showerror("Error", "Failed to save student data")
                self._reset_registration()
                return
            
            # Add face encoding
            success = self.database_manager.add_face_encoding(student_id, averaged_encoding, 'simple')
            if not success:
                messagebox.showerror("Error", "Failed to save face encoding")
                self._reset_registration()
                return
            
            # Show success message
            messagebox.showinfo("Success", 
                               f"Student {student_id} registered successfully!\n\n"
                               f"Captured {len(self.captured_encodings)} face encodings\n"
                               f"Saved as pickle-based encoding")
            
            # Close dialog
            self.dialog.destroy()
            
        except Exception as e:
            print(f"Error completing registration: {e}")
            messagebox.showerror("Error", f"Failed to complete registration: {str(e)}")
            self._reset_registration()
    
    def _reset_registration(self):
        """Reset registration state"""
        self.is_registering = False
        self.registration_progress = 0
        self.captured_encodings = []
        
        self.start_button.config(state='normal')
        self.student_id_entry.config(state='normal')
        self.progress_var.set(0)
        self.progress_label.config(text="Ready to start registration")
        self.status_label.config(text="Position your face in the camera", foreground='blue')
    
    def _cancel(self):
        """Cancel registration and close dialog"""
        self.is_registering = False
        self.camera_running = False
        self.dialog.destroy()
    
    def _is_valid_face_region(self, x, y, w, h, frame):
        """
        Validate if a detected region is actually a face
        
        Args:
            x, y, w, h: Face bounding box coordinates
            frame: Original frame for additional analysis
            
        Returns:
            bool: True if region appears to be a valid face
        """
        try:
            # Check 1: Aspect ratio - faces are roughly square or slightly taller
            aspect_ratio = w / h
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:  # Too wide or too tall
                return False
            
            # Check 2: Size - face should be reasonably sized
            face_area = w * h
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area
            
            if face_ratio < 0.01 or face_ratio > 0.3:  # Too small or too large
                return False
            
            # Check 3: Position - face should be in upper portion of frame
            frame_height = frame.shape[0]
            face_center_y = y + h // 2
            if face_center_y > frame_height * 0.7:  # Face too low in frame
                return False
            
            # Check 4: Extract face region and analyze
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                return False
            
            # Check 5: Color distribution - faces have specific color patterns
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Check for reasonable brightness variation (faces have shadows/highlights)
            mean_brightness = np.mean(face_gray)
            std_brightness = np.std(face_gray)
            
            if mean_brightness < 30 or mean_brightness > 200:  # Too dark or too bright
                return False
            
            if std_brightness < 15:  # Too uniform (like clothing)
                return False
            
            # Check 6: Edge density - faces have more edges than clothing
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            
            if edge_density < 0.05:  # Too few edges (like plain clothing)
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in face validation: {e}")
            return False
    
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
