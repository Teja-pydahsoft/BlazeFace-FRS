"""
Student Re-registration Tool for BlazeFace-FRS
Handles re-registration of problematic students with enhanced quality control
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from ..core.database import DatabaseManager
from ..core.simple_face_embedder import SimpleFaceEmbedder
from ..core.insightface_embedder import InsightFaceEmbedder
from ..core.standard_face_embedder import StandardFaceEmbedder
from ..utils.encoding_quality_checker import EncodingQualityChecker
from ..utils.camera_utils import CameraManager

class StudentReregistrationTool:
    def __init__(self, parent, database_manager: DatabaseManager, config: Dict[str, Any]):
        """
        Initialize student re-registration tool
        
        Args:
            parent: Parent window
            database_manager: Database manager instance
            config: Application configuration
        """
        self.parent = parent
        self.database_manager = database_manager
        self.config = config
        
        # Initialize components
        self.quality_checker = EncodingQualityChecker(database_manager)
        self.camera_manager = None
        self.face_detector = None
        self.embedder = None
        
        # UI state
        self.current_student_id = None
        self.captured_faces = []
        self.is_capturing = False
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Student Re-registration Tool - BlazeFace-FRS")
        self.dialog.geometry("1000x700")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Setup UI
        self._setup_ui()
        
        # Center dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Setup the re-registration UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top panel - Student Selection
        self._setup_student_selection(main_frame)
        
        # Middle panel - Camera and Capture
        self._setup_capture_panel(main_frame)
        
        # Bottom panel - Quality Control and Actions
        self._setup_quality_panel(main_frame)
    
    def _setup_student_selection(self, parent):
        """Setup student selection panel"""
        selection_frame = ttk.LabelFrame(parent, text="Student Selection", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Student ID selection
        ttk.Label(selection_frame, text="Student ID:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.student_id_var = tk.StringVar()
        student_id_combo = ttk.Combobox(selection_frame, textvariable=self.student_id_var, width=15)
        student_id_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Load problematic students
        self._load_problematic_students(student_id_combo)
        
        ttk.Button(selection_frame, text="Load Student Info", 
                  command=self._load_student_info).pack(side=tk.LEFT, padx=(0, 10))
        
        # Student info display
        self.student_info_label = ttk.Label(selection_frame, text="No student selected")
        self.student_info_label.pack(side=tk.LEFT, padx=(20, 0))
    
    def _setup_capture_panel(self, parent):
        """Setup camera capture panel"""
        capture_frame = ttk.LabelFrame(parent, text="Face Capture with Quality Control", padding="10")
        capture_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Camera preview frame
        camera_frame = ttk.Frame(capture_frame)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(camera_frame, width=640, height=480, bg='black')
        self.camera_canvas.pack(side=tk.LEFT, padx=(0, 10))
        
        # Instructions and controls
        controls_frame = ttk.Frame(camera_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Instructions
        instructions = ttk.Label(controls_frame, text="""IMPROVED CAPTURE INSTRUCTIONS:

1. Ensure EXCELLENT lighting
   - Face well-lit from front
   - No shadows on face
   - Avoid backlighting

2. Position face correctly
   - Face centered in view
   - Looking directly at camera
   - Full face visible (no tilting)

3. Capture multiple angles
   - Front view (required)
   - Slight left angle
   - Slight right angle

4. Quality requirements
   - Clear, sharp image
   - Good contrast
   - No blur or motion

5. Click 'Start Camera' first
   - Verify good positioning
   - Check lighting quality
   - Then capture faces""", justify=tk.LEFT)
        instructions.pack(fill=tk.X, pady=(0, 10))
        
        # Camera controls
        ttk.Button(controls_frame, text="Start Camera", 
                  command=self._start_camera).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Stop Camera", 
                  command=self._stop_camera).pack(fill=tk.X, pady=2)
        
        # Capture controls
        ttk.Separator(controls_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="Face Capture:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Button(controls_frame, text="Capture Face", 
                  command=self._capture_face).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Clear Captures", 
                  command=self._clear_captures).pack(fill=tk.X, pady=2)
        
        # Quality display
        ttk.Separator(controls_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.quality_label = ttk.Label(controls_frame, text="Quality: Not Captured")
        self.quality_label.pack(anchor=tk.W)
        
        self.capture_count_label = ttk.Label(controls_frame, text="Captures: 0")
        self.capture_count_label.pack(anchor=tk.W)
    
    def _setup_quality_panel(self, parent):
        """Setup quality control panel"""
        quality_frame = ttk.LabelFrame(parent, text="Quality Control & Actions", padding="10")
        quality_frame.pack(fill=tk.X)
        
        # Quality analysis
        analysis_frame = ttk.Frame(quality_frame)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(analysis_frame, text="Analyze Current Captures", 
                  command=self._analyze_captures).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(analysis_frame, text="Compare with Existing", 
                  command=self._compare_with_existing).pack(side=tk.LEFT, padx=(0, 10))
        
        # Results display
        self.analysis_text = tk.Text(quality_frame, height=8, width=80)
        self.analysis_text.pack(fill=tk.X, pady=(0, 10))
        
        # Action buttons
        action_frame = ttk.Frame(quality_frame)
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Re-register Student", 
                  command=self._reregister_student).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="Delete Old Encodings", 
                  command=self._delete_old_encodings).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="Close", 
                  command=self._close_dialog).pack(side=tk.RIGHT)
    
    def _load_problematic_students(self, combo):
        """Load problematic students into combobox"""
        try:
            # Get students with face encodings
            students = self.database_manager.get_all_students()
            student_ids = [s['student_id'] for s in students if s['student_id']]
            
            # Prioritize problematic students
            problematic = ['1233', '52856']
            prioritized = [sid for sid in problematic if sid in student_ids]
            prioritized.extend([sid for sid in student_ids if sid not in problematic])
            
            combo['values'] = prioritized
            if prioritized:
                combo.set(prioritized[0])
                self._load_student_info()
                
        except Exception as e:
            print(f"Error loading students: {e}")
    
    def _load_student_info(self):
        """Load selected student information"""
        try:
            student_id = self.student_id_var.get().strip()
            if not student_id:
                return
            
            student = self.database_manager.get_student(student_id)
            if student:
                info = f"Name: {student['name']} | Dept: {student.get('department', 'N/A')} | Year: {student.get('year', 'N/A')}"
                self.student_info_label.config(text=info)
                self.current_student_id = student_id
            else:
                self.student_info_label.config(text="Student not found")
                self.current_student_id = None
                
        except Exception as e:
            print(f"Error loading student info: {e}")
    
    def _start_camera(self):
        """Start camera preview"""
        try:
            if self.camera_manager:
                self.camera_manager.stop_camera()
            
            camera_source = self.config.get('camera_index', 0)
            self.camera_manager = CameraManager(camera_source)
            
            if self.camera_manager.start_camera():
                self.is_capturing = True
                self._update_camera_preview()
            else:
                messagebox.showerror("Error", "Failed to start camera")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def _stop_camera(self):
        """Stop camera preview"""
        try:
            self.is_capturing = False
            if self.camera_manager:
                self.camera_manager.stop_camera()
                self.camera_canvas.delete("all")
                self.camera_canvas.create_text(320, 240, text="Camera Stopped", 
                                             fill='white', font=('Arial', 16))
        except Exception as e:
            print(f"Error stopping camera: {e}")
    
    def _update_camera_preview(self):
        """Update camera preview"""
        try:
            if not self.is_capturing or not self.camera_manager:
                return
            
            frame = self.camera_manager.get_frame()
            if frame is not None:
                # Resize frame to fit canvas
                height, width = frame.shape[:2]
                canvas_width = self.camera_canvas.winfo_width()
                canvas_height = self.camera_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    scale = min(canvas_width / width, canvas_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    resized = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PhotoImage
                    photo = self._cv2_to_photo(rgb_frame)
                    
                    # Clear canvas and display image
                    self.camera_canvas.delete("all")
                    self.camera_canvas.create_image(canvas_width//2, canvas_height//2, 
                                                  image=photo, anchor=tk.CENTER)
            
            # Schedule next update
            if self.is_capturing:
                self.dialog.after(30, self._update_camera_preview)
                
        except Exception as e:
            print(f"Error updating camera preview: {e}")
            if self.is_capturing:
                self.dialog.after(1000, self._update_camera_preview)
    
    def _cv2_to_photo(self, cv2_image):
        """Convert OpenCV image to PhotoImage"""
        try:
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(cv2_image)
            return ImageTk.PhotoImage(pil_image)
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    def _capture_face(self):
        """Capture face with quality control"""
        try:
            if not self.is_capturing or not self.camera_manager:
                messagebox.showwarning("Warning", "Please start camera first")
                return
            
            if not self.current_student_id:
                messagebox.showwarning("Warning", "Please select a student first")
                return
            
            frame = self.camera_manager.get_frame()
            if frame is None:
                messagebox.showerror("Error", "No camera frame available")
                return
            
            # Detect face
            face_region = self._detect_face(frame)
            if face_region is None:
                messagebox.showwarning("Warning", "No face detected. Please position face clearly in camera view.")
                return
            
            # Extract face
            face_image = frame[face_region[1]:face_region[3], face_region[0]:face_region[2]]
            
            # Check face quality
            quality_score, quality_msg = self._check_face_quality(face_image)
            
            if quality_score < 0.7:
                messagebox.showwarning("Low Quality", f"Face quality too low: {quality_msg}\nPlease improve lighting and positioning.")
                return
            
            # Generate embedding
            if not self.embedder:
                self.embedder = SimpleFaceEmbedder()
            
            embedding = self.embedder.get_embedding(face_image)
            if embedding is None:
                messagebox.showerror("Error", "Failed to generate face embedding")
                return
            
            # Store capture
            capture_data = {
                'image': face_image,
                'embedding': embedding,
                'quality_score': quality_score,
                'timestamp': time.time(),
                'face_region': face_region
            }
            
            self.captured_faces.append(capture_data)
            
            # Update UI
            self.quality_label.config(text=f"Quality: {quality_score:.2f} - {quality_msg}")
            self.capture_count_label.config(text=f"Captures: {len(self.captured_faces)}")
            
            messagebox.showinfo("Success", f"Face captured successfully!\nQuality: {quality_score:.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture face: {str(e)}")
    
    def _detect_face(self, frame):
        """Detect face in frame"""
        try:
            # Use OpenCV face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                return (x, y, x + w, y + h)
            
            return None
            
        except Exception as e:
            print(f"Error detecting face: {e}")
            return None
    
    def _check_face_quality(self, face_image):
        """Check face image quality"""
        try:
            # Basic quality checks
            height, width = face_image.shape[:2]
            
            # Size check
            if width < 100 or height < 100:
                return 0.0, "Face too small"
            
            # Brightness check
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness < 50:
                return 0.3, "Too dark"
            elif brightness > 200:
                return 0.3, "Too bright"
            
            # Contrast check
            contrast = np.std(gray)
            if contrast < 20:
                return 0.4, "Low contrast"
            
            # Blur check (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:
                return 0.5, "Blurry image"
            
            # Calculate overall quality score
            size_score = min(1.0, (width * height) / (150 * 150))
            brightness_score = 1.0 - abs(brightness - 128) / 128
            contrast_score = min(1.0, contrast / 50)
            blur_score_normalized = min(1.0, blur_score / 500)
            
            overall_score = (size_score + brightness_score + contrast_score + blur_score_normalized) / 4
            
            if overall_score > 0.8:
                quality_msg = "Excellent"
            elif overall_score > 0.7:
                quality_msg = "Good"
            elif overall_score > 0.5:
                quality_msg = "Acceptable"
            else:
                quality_msg = "Poor"
            
            return overall_score, quality_msg
            
        except Exception as e:
            print(f"Error checking face quality: {e}")
            return 0.0, "Quality check failed"
    
    def _clear_captures(self):
        """Clear all captured faces"""
        self.captured_faces.clear()
        self.quality_label.config(text="Quality: Not Captured")
        self.capture_count_label.config(text="Captures: 0")
        self.analysis_text.delete(1.0, tk.END)
    
    def _analyze_captures(self):
        """Analyze captured faces"""
        try:
            if not self.captured_faces:
                messagebox.showwarning("Warning", "No faces captured yet")
                return
            
            analysis = "FACE CAPTURE ANALYSIS:\n" + "="*50 + "\n\n"
            
            for i, capture in enumerate(self.captured_faces):
                analysis += f"Capture {i+1}:\n"
                analysis += f"  Quality Score: {capture['quality_score']:.3f}\n"
                analysis += f"  Face Size: {capture['image'].shape[1]}x{capture['image'].shape[0]}\n"
                analysis += f"  Embedding Norm: {np.linalg.norm(capture['embedding']):.3f}\n"
                
                # Check internal consistency
                if i > 0:
                    similarity = np.dot(capture['embedding'], self.captured_faces[0]['embedding'])
                    analysis += f"  Similarity to Capture 1: {similarity:.3f}\n"
                
                analysis += "\n"
            
            # Overall assessment
            avg_quality = np.mean([c['quality_score'] for c in self.captured_faces])
            analysis += f"Average Quality: {avg_quality:.3f}\n"
            
            if avg_quality > 0.8:
                analysis += "✅ EXCELLENT quality - Ready for registration\n"
            elif avg_quality > 0.7:
                analysis += "✅ GOOD quality - Suitable for registration\n"
            elif avg_quality > 0.5:
                analysis += "⚠️  ACCEPTABLE quality - Consider re-capturing\n"
            else:
                analysis += "❌ POOR quality - Re-capture required\n"
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, analysis)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze captures: {str(e)}")
    
    def _compare_with_existing(self):
        """Compare captures with existing encodings"""
        try:
            if not self.captured_faces:
                messagebox.showwarning("Warning", "No faces captured yet")
                return
            
            if not self.current_student_id:
                messagebox.showwarning("Warning", "No student selected")
                return
            
            # Get existing encodings
            existing_encodings = self.database_manager.get_face_encodings(self.current_student_id)
            
            if not existing_encodings:
                messagebox.showinfo("Info", "No existing encodings found for this student")
                return
            
            comparison = "COMPARISON WITH EXISTING ENCODINGS:\n" + "="*50 + "\n\n"
            
            for i, capture in enumerate(self.captured_faces):
                comparison += f"Capture {i+1} vs Existing Encodings:\n"
                
                similarities = []
                for j, (_, existing_encoding, _, _) in enumerate(existing_encodings):
                    similarity = np.dot(capture['embedding'], existing_encoding)
                    similarities.append(similarity)
                    comparison += f"  vs Encoding {j+1}: {similarity:.3f}\n"
                
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                comparison += f"  Average Similarity: {avg_similarity:.3f}\n"
                comparison += f"  Maximum Similarity: {max_similarity:.3f}\n"
                
                if avg_similarity > 0.8:
                    comparison += "  ✅ GOOD consistency with existing data\n"
                elif avg_similarity > 0.6:
                    comparison += "  ⚠️  MODERATE consistency - may be acceptable\n"
                else:
                    comparison += "  ❌ LOW consistency - significant difference\n"
                
                comparison += "\n"
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, comparison)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare with existing: {str(e)}")
    
    def _reregister_student(self):
        """Re-register student with new captures"""
        try:
            if not self.captured_faces:
                messagebox.showwarning("Warning", "No faces captured yet")
                return
            
            if not self.current_student_id:
                messagebox.showwarning("Warning", "No student selected")
                return
            
            # Confirm re-registration
            result = messagebox.askyesno("Confirm Re-registration", 
                                       f"Are you sure you want to re-register student {self.current_student_id}?\n\n"
                                       f"This will replace all existing face encodings with the {len(self.captured_faces)} new captures.")
            
            if not result:
                return
            
            # Delete old encodings
            self._delete_old_encodings(silent=True)
            
            # Add new encodings
            success_count = 0
            for i, capture in enumerate(self.captured_faces):
                # Save face image
                timestamp = int(time.time() * 1000) + i
                filename = f"{self.current_student_id}_{timestamp}.jpg"
                filepath = os.path.join(self.config.get('face_data_path', 'face_data'), filename)
                
                cv2.imwrite(filepath, capture['image'])
                
                # Add encoding to database
                success = self.database_manager.add_face_encoding(
                    self.current_student_id,
                    capture['embedding'],
                    'simple',
                    filepath
                )
                
                if success:
                    success_count += 1
            
            if success_count > 0:
                messagebox.showinfo("Success", 
                                  f"Student {self.current_student_id} re-registered successfully!\n"
                                  f"Added {success_count} new face encodings.")
                
                # Clear captures
                self._clear_captures()
            else:
                messagebox.showerror("Error", "Failed to add any new encodings")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to re-register student: {str(e)}")
    
    def _delete_old_encodings(self, silent=False):
        """Delete old encodings for current student"""
        try:
            if not self.current_student_id:
                if not silent:
                    messagebox.showwarning("Warning", "No student selected")
                return
            
            if not silent:
                result = messagebox.askyesno("Confirm Deletion", 
                                           f"Delete all existing face encodings for student {self.current_student_id}?")
                if not result:
                    return
            
            # Delete encodings from database
            with self.database_manager.db_path as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM face_encodings WHERE student_id = ?", (self.current_student_id,))
                conn.commit()
                
            if not silent:
                messagebox.showinfo("Success", "Old encodings deleted successfully")
                
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Failed to delete old encodings: {str(e)}")
    
    def _close_dialog(self):
        """Close the dialog"""
        try:
            self._stop_camera()
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
