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

from ..core.insightface_embedder import InsightFaceEmbedder
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
        self.face_embedder = InsightFaceEmbedder(model_name='buffalo_l')  # Use ArcFace/InsightFace for all encodings
        
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
        
        self.upload_btn = ttk.Button(capture_frame, text="Upload Image",
                                    command=self._upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=(5, 0))
        
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

    def _check_for_duplicate_face(self, new_embedding: np.ndarray) -> Optional[str]:
        """Check if a face embedding already exists in the database."""
        all_encodings = self.database_manager.get_face_encodings()
        for student_id, existing_encoding, _, _ in all_encodings:
            distance = np.linalg.norm(new_embedding - existing_encoding)
            if distance < 0.6:  # Threshold for considering faces as the same
                return student_id
        return None

    def _capture_face(self):
        """Capture face from current frame"""
        try:
            if self.current_frame is None:
                messagebox.showwarning("Warning", "No camera feed available. Please wait for camera to initialize.")
                return
            # Debug: Print frame info
            print(f"[DEBUG] Frame shape: {self.current_frame.shape}, dtype: {self.current_frame.dtype}")
            print(f"[DEBUG] Frame sample pixel: {self.current_frame[0,0]}")
            # Ensure frame is RGB and uint8
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB) if self.current_frame.shape[2] == 3 else self.current_frame
            frame_rgb = frame_rgb.astype(np.uint8)
            # Detect faces
            faces = self.face_detector.detect_faces(frame_rgb)
            print(f"[DEBUG] BlazeFace detected {len(faces)} faces: {faces}")
            if not faces:
                messagebox.showwarning("Warning", "No face detected in current frame.\n\nPlease ensure:\n- Your face is clearly visible\n- Good lighting conditions\n- Face is centered in the camera view")
                return
            face_box = faces[0]
            x, y, w, h, confidence = face_box
            if confidence < 0.5:
                messagebox.showwarning("Warning", f"Face detection confidence too low: {confidence:.2f}\n\nPlease ensure better lighting and face positioning.")
                return
            # Pass full RGB frame to InsightFace
            insightface_results = self.face_embedder.detect_and_encode_faces(frame_rgb)
            print(f"[DEBUG] InsightFace detected {len(insightface_results)} faces: {[f[0]['bbox'] for f in insightface_results]}")
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
            blaze_bbox = [x, y, x + w, y + h]
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
                print(f"[ERROR] InsightFace could not find a matching face. Saving frame for inspection.")
                import uuid
                fname = f"failed_insightface_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(fname, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                print(f"[ERROR] Saved problematic frame to {fname}")
                messagebox.showerror("Error", "InsightFace could not find a matching face in the frame. Please try again with a clearer face.")
                return
            embedding = best_embedding
            print(f"[DEBUG] Selected embedding with IOU={best_iou:.2f}")
            print(f"[DEBUG] Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"[DEBUG] Embedding first 5 values: {embedding[:5]}")
            student_id = self._check_for_duplicate_face(embedding)
            if student_id:
                messagebox.showinfo("Duplicate Face Detected", 
                                   f"A face similar to the one being captured is already registered for Student ID: {student_id}\n\nCapture anyway?", 
                                   icon=messagebox.WARNING)
            self.captured_faces.append({'embedding': embedding, 'confidence': confidence, 'timestamp': time.time()})
            self._update_captured_faces_display()
            self.status_label.config(text=f"Status: Face captured - {len(self.captured_faces)} face(s) saved", 
                                   foreground='green')
        except Exception as e:
            print(f"Error capturing face: {e}")
            messagebox.showerror("Error", f"An error occurred while capturing face: {e}")
    
    def _update_captured_faces_display(self):
        """Update the display of captured faces"""
        try:
            # Clear existing images
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            # Display each captured face
            for idx, (face_region, _) in enumerate(self.captured_faces):
                # Resize face region for display
                face_display = cv2.resize(face_region, (100, 100))
                
                # Convert to PhotoImage
                face_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(face_display, cv2.COLOR_BGR2RGB)))
                
                # Create label for face image
                label = ttk.Label(self.scrollable_frame, image=face_image)
                label.image = face_image  # Keep reference to prevent garbage collection
                label.grid(row=0, column=idx, padx=5, pady=5)
            
            # Update scroll region
            self.scrollable_frame.update_idletasks()
        except Exception as e:
            print(f"Error updating captured faces display: {e}")
    
    def _register_student(self):
        """Register the student with captured faces"""
        try:
            student_id = self.student_id_var.get().strip()
            name = self.name_var.get().strip()
            email = self.email_var.get().strip()
            phone = self.phone_var.get().strip()
            department = self.department_var.get().strip()
            year = self.year_var.get().strip()
            
            # Validate required fields
            if not student_id or not name:
                messagebox.showwarning("Warning", "Student ID and Name are required fields.")
                return
            
            # Check if student ID already exists
            if self.database_manager.get_student_by_id(student_id):
                messagebox.showwarning("Warning", f"Student ID {student_id} is already registered.")
                return
            
            # Check for duplicate faces in captured faces
            if len(self.captured_faces) == 0:
                messagebox.showwarning("Warning", "No faces captured. Please capture at least one face before registering.")
                return
            
            # Register student in database
            self.database_manager.register_student(student_id, name, email, phone, department, year)
            
            # Encode and save captured faces
            for idx, (face_region, embedding) in enumerate(self.captured_faces):
                # Save face encoding to database
                self.database_manager.save_face_encoding(student_id, embedding)
                
                # Optionally, save face image to file system
                if self.config.get('save_face_images', False):
                    self._save_face_image_to_file_system(student_id, face_region, idx)
            
            messagebox.showinfo("Success", "Student registered successfully!")
            
            # Close dialog
            self.dialog.destroy()
        except Exception as e:
            print(f"Error registering student: {e}")
            messagebox.showerror("Error", f"An error occurred while registering student: {e}")
    
    def _save_face_image_to_file_system(self, student_id: str, face_region: np.ndarray, index: int):
        """Save face image to file system"""
        try:
            # Create directory for student if not exists
            student_dir = os.path.join(self.config.get('face_images_directory', 'registered_faces'), student_id)
            os.makedirs(student_dir, exist_ok=True)
            
            # Save face image
            image_path = os.path.join(student_dir, f"face_{index+1}.jpg")
            cv2.imwrite(image_path, face_region)
            
            print(f"Saved face image to: {image_path}")
        except Exception as e:
            print(f"Error saving face image to file system: {e}")
    
    def _debug_camera(self):
        """Debug camera settings and functionality"""
        try:
            if self.camera_manager.is_camera_available():
                # Release camera if already started
                if self.is_capturing:
                    self.camera_manager.stop_camera()
                    self.is_capturing = False
                
                # Open debug window
                cv2.namedWindow("Camera Debug", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Camera Debug", 800, 600)
                
                # Start camera preview in debug window
                self.camera_manager.start_camera_preview(window_name="Camera Debug")
                
                # Show debug information on window
                cv2.putText(self.camera_manager.get_current_frame(), "Camera Debug Mode - Press 'q' to exit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                while self.is_capturing:
                    # Update camera frame
                    frame = self.camera_manager.get_current_frame()
                    if frame is not None:
                        cv2.imshow("Camera Debug", frame)
                    
                    # Check for 'q' key press to exit debug mode
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Stop camera and close debug window
                self.camera_manager.stop_camera()
                cv2.destroyWindow("Camera Debug")
            else:
                messagebox.showwarning("Warning", "Camera not available for debugging.")
        except Exception as e:
            print(f"Error in debug camera: {e}")
            messagebox.showerror("Error", f"An error occurred in debug camera: {e}")
    
    def _cancel(self):
        """Cancel registration and close dialog"""
        if messagebox.askokcancel("Cancel", "Are you sure you want to cancel registration?"):
            # Release camera if allocated
            if self.camera_manager and self.camera_manager.is_camera_available():
                self.camera_manager.stop_camera()
            
            # Close dialog
            self.dialog.destroy()
    
    def _center_dialog(self):
        """Center the dialog on the screen"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")