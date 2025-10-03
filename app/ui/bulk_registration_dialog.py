"""
Bulk Registration Dialog for BlazeFace-FRS
Handles bulk student registration with CSV upload and photo management
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import threading
import time
from datetime import datetime
import json

from ..core.database import DatabaseManager
from ..core.simple_face_embedder import SimpleFaceEmbedder
from ..core.blazeface_detector import BlazeFaceDetector
from ..utils.encoding_quality_checker import EncodingQualityChecker

class BulkRegistrationDialog:
    def __init__(self, parent, database_manager: DatabaseManager, config: Dict[str, Any]):
        """
        Initialize bulk registration dialog
        
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
        self.quality_checker = EncodingQualityChecker(database_manager)
        self.embedder = None
        
        # Data storage
        self.students_data = []
        self.photo_files = {}
        self.processing_results = []
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Bulk Student Registration - BlazeFace-FRS")
        self.dialog.geometry("1200x800")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Setup UI
        self._setup_ui()
        
        # Center dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Setup the bulk registration UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top panel - File upload
        self._setup_file_upload_panel(main_frame)
        
        # Middle panel - Data preview and photo management
        self._setup_data_preview_panel(main_frame)
        
        # Bottom panel - Processing and results
        self._setup_processing_panel(main_frame)
    
    def _setup_file_upload_panel(self, parent):
        """Setup file upload panel"""
        upload_frame = ttk.LabelFrame(parent, text="Step 1: Upload Files", padding="10")
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        # CSV upload section
        csv_frame = ttk.Frame(upload_frame)
        csv_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(csv_frame, text="CSV File:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.csv_path_var = tk.StringVar()
        csv_entry = ttk.Entry(csv_frame, textvariable=self.csv_path_var, width=50, state='readonly')
        csv_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(csv_frame, text="Browse CSV", command=self._browse_csv_file).pack(side=tk.LEFT, padx=(0, 10))
        
        # Photo directory section
        photo_frame = ttk.Frame(upload_frame)
        photo_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(photo_frame, text="Photos Directory:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.photo_dir_var = tk.StringVar()
        photo_entry = ttk.Entry(photo_frame, textvariable=self.photo_dir_var, width=50, state='readonly')
        photo_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(photo_frame, text="Browse Directory", command=self._browse_photo_directory).pack(side=tk.LEFT, padx=(0, 10))
        
        # Load data button
        ttk.Button(upload_frame, text="Load Data", command=self._load_data, 
                  style='Accent.TButton').pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(upload_frame, 
                               text="Instructions:\n" 
                                    "1. Upload CSV file with student data (columns: student_id, name, email, phone, department, year)\n" 
                                    "2. Select directory containing student photos (named as student_id.jpg or student_id.png)\n" 
                                    "3. Click 'Load Data' to preview the data before processing",
                               font=('Arial', 9), foreground='#7f8c8d')
        instructions.pack(pady=(10, 0))
    
    def _setup_data_preview_panel(self, parent):
        """Setup data preview panel"""
        preview_frame = ttk.LabelFrame(parent, text="Step 2: Preview Data", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(preview_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Students data tab
        students_frame = ttk.Frame(notebook)
        notebook.add(students_frame, text="Students Data")
        
        # Students table
        columns = ('Student ID', 'Name', 'Email', 'Phone', 'Department', 'Year', 'Photo Status', 'Quality')
        self.students_tree = ttk.Treeview(students_frame, columns=columns, show='headings', height=10)
        
        # Configure columns
        column_widths = {
            'Student ID': 100, 'Name': 150, 'Email': 200, 'Phone': 120,
            'Department': 120, 'Year': 80, 'Photo Status': 100, 'Quality': 80
        }
        
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=column_widths[col], anchor=tk.CENTER)
        
        # Scrollbars for students table
        students_v_scroll = ttk.Scrollbar(students_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        students_h_scroll = ttk.Scrollbar(students_frame, orient=tk.HORIZONTAL, command=self.students_tree.xview)
        self.students_tree.configure(yscrollcommand=students_v_scroll.set, xscrollcommand=students_h_scroll.set)
        
        # Pack students table
        self.students_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        students_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        students_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Photo preview tab
        photo_frame = ttk.Frame(notebook)
        notebook.add(photo_frame, text="Photo Preview")
        
        # Photo selection and preview
        photo_selection_frame = ttk.Frame(photo_frame)
        photo_selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(photo_selection_frame, text="Select Student:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.photo_student_var = tk.StringVar()
        self.photo_student_combo = ttk.Combobox(photo_selection_frame, textvariable=self.photo_student_var, 
                                               state='readonly', width=20)
        self.photo_student_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.photo_student_combo.bind('<<ComboboxSelected>>', self._preview_photo)
        
        # Photo preview canvas
        self.photo_canvas = tk.Canvas(photo_frame, width=400, height=300, bg='lightgray')
        self.photo_canvas.pack(pady=10)
        
        # Photo info
        self.photo_info_label = ttk.Label(photo_frame, text="No photo selected")
        self.photo_info_label.pack(pady=5)
    
    def _setup_processing_panel(self, parent):
        """Setup processing panel"""
        processing_frame = ttk.LabelFrame(parent, text="Step 3: Process Registration", padding="10")
        processing_frame.pack(fill=tk.X)
        
        # Processing controls
        controls_frame = ttk.Frame(processing_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Embedder selection
        ttk.Label(controls_frame, text="Face Encoder:").pack(side=tk.LEFT, padx=(0, 10))
        self.embedder_var = tk.StringVar(value="simple")
        embedder_combo = ttk.Combobox(controls_frame, textvariable=self.embedder_var,
                                     values=["simple"], state='readonly', width=15)
        embedder_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        # Quality threshold
        ttk.Label(controls_frame, text="Quality Threshold:").pack(side=tk.LEFT, padx=(0, 10))
        self.quality_threshold_var = tk.StringVar(value="0.7")
        quality_spin = ttk.Spinbox(controls_frame, from_=0.5, to=1.0, increment=0.1,
                                  textvariable=self.quality_threshold_var, width=8)
        quality_spin.pack(side=tk.LEFT, padx=(0, 20))
        
        # Process button
        self.process_btn = ttk.Button(controls_frame, text="Process All Students", 
                                     command=self._process_all_students, style='Accent.TButton')
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(processing_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.pack(pady=(0, 10))
        
        # Results display
        results_frame = ttk.LabelFrame(processing_frame, text="Processing Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, height=8, width=80)
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Close button
        ttk.Button(processing_frame, text="Close", command=self._close_dialog).pack(side=tk.RIGHT, pady=(10, 0))
    
    def _browse_csv_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path_var.set(filename)
    
    def _browse_photo_directory(self):
        """Browse for photo directory"""
        directory = filedialog.askdirectory(title="Select Photos Directory")
        if directory:
            self.photo_dir_var.set(directory)
    
    def _load_data(self):
        """Load and preview data from CSV and photos"""
        try:
            csv_path = self.csv_path_var.get()
            photo_dir = self.photo_dir_var.get()
            
            if not csv_path or not photo_dir:
                messagebox.showwarning("Warning", "Please select both CSV file and photos directory")
                return
            
            # Load CSV data
            self.students_data = self._load_csv_data(csv_path)
            
            # Load photo files
            self.photo_files = self._load_photo_files(photo_dir)
            
            # Update preview
            self._update_students_preview()
            self._update_photo_preview()
            
            messagebox.showinfo("Success", f"Loaded {len(self.students_data)} students and {len(self.photo_files)} photos")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def _load_csv_data(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load student data from CSV file"""
        try:
            students = []
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Validate header
                required_columns = ['student_id', 'name']
                if not all(col in reader.fieldnames for col in required_columns):
                    missing = [col for col in required_columns if col not in reader.fieldnames]
                    raise Exception(f"CSV file is missing required columns: {', '.join(missing)}")

                for row in reader:
                    student = {
                        'student_id': row.get('student_id', '').strip(),
                        'name': row.get('name', '').strip(),
                        'email': row.get('email', '').strip(),
                        'phone': row.get('phone', '').strip(),
                        'department': row.get('department', '').strip(),
                        'year': row.get('year', '').strip()
                    }
                    students.append(student)
            return students
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
    
    def _load_photo_files(self, photo_dir: str) -> Dict[str, str]:
        """Load photo files from directory"""
        try:
            photo_files = {}
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
            
            for filename in os.listdir(photo_dir):
                if any(filename.lower().endswith(ext) for ext in supported_formats):
                    # Extract student ID from filename (remove extension)
                    student_id = os.path.splitext(filename)[0]
                    filepath = os.path.join(photo_dir, filename)
                    photo_files[student_id] = filepath
            
            return photo_files
        except Exception as e:
            raise Exception(f"Error loading photo files: {str(e)}")
    
    def _update_students_preview(self):
        """Update students data preview"""
        try:
            # Clear existing items
            for item in self.students_tree.get_children():
                self.students_tree.delete(item)
            
            # Add students to tree
            for student in self.students_data:
                student_id = student['student_id']
                
                # Check photo status
                photo_status = "Found" if student_id in self.photo_files else "Missing"
                
                # Check if student already exists in database
                existing_student = self.database_manager.get_student(student_id)
                if existing_student:
                    photo_status += " (Exists)"
                
                # Estimate quality (will be updated after processing)
                quality = "Pending"
                
                self.students_tree.insert('', 'end', values=(
                    student_id,
                    student['name'],
                    student['email'],
                    student['phone'],
                    student['department'],
                    student['year'],
                    photo_status,
                    quality
                ))
            
        except Exception as e:
            print(f"Error updating students preview: {e}")
    
    def _update_photo_preview(self):
        """Update photo preview options"""
        try:
            student_ids = [student['student_id'] for student in self.students_data]
            self.photo_student_combo['values'] = student_ids
            if student_ids:
                self.photo_student_combo.set(student_ids[0])
                self._preview_photo()
        except Exception as e:
            print(f"Error updating photo preview: {e}")
    
    def _preview_photo(self, event=None):
        """Preview selected student's photo"""
        try:
            student_id = self.photo_student_var.get()
            if not student_id or student_id not in self.photo_files:
                self.photo_canvas.delete("all")
                self.photo_canvas.create_text(200, 150, text="No photo available", 
                                            fill='gray', font=('Arial', 14))
                self.photo_info_label.config(text="No photo selected")
                return
            
            photo_path = self.photo_files[student_id]
            
            # Load and display image
            image = cv2.imread(photo_path)
            if image is None:
                self.photo_canvas.delete("all")
                self.photo_canvas.create_text(200, 150, text="Invalid image file", 
                                            fill='red', font=('Arial', 14))
                return
            
            # Resize image to fit canvas
            canvas_width = 400
            canvas_height = 300
            
            height, width = image.shape[:2]
            scale = min(canvas_width / width, canvas_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image, (new_width, new_height))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            photo = self._cv2_to_photo(rgb_image)
            
            # Display image
            self.photo_canvas.delete("all")
            self.photo_canvas.create_image(canvas_width//2, canvas_height//2, 
                                         image=photo, anchor=tk.CENTER)
            
            # Update info
            info_text = f"Student: {student_id}\nSize: {width}x{height}\nFile: {os.path.basename(photo_path)}"
            self.photo_info_label.config(text=info_text)
            
        except Exception as e:
            print(f"Error previewing photo: {e}")
    
    def _cv2_to_photo(self, cv2_image):
        """Convert OpenCV image to PhotoImage"""
        try:
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(cv2_image)
            return ImageTk.PhotoImage(pil_image)
        except Exception as e:
            print(f"Error converting image: {e}")
            return None

    def _check_for_duplicate_face(self, new_embedding: np.ndarray) -> Optional[str]:
        """Check if a face embedding already exists in the database."""
        all_encodings = self.database_manager.get_face_encodings()
        for student_id, existing_encoding, _, _ in all_encodings:
            distance = np.linalg.norm(new_embedding - existing_encoding)
            if distance < 0.6:  # Threshold for considering faces as the same
                return student_id
        return None

    def _process_all_students(self):
        """Process all students for registration"""
        try:
            if not self.students_data:
                messagebox.showwarning("Warning", "No student data loaded")
                return
            
            # Initialize embedder
            embedder_type = self.embedder_var.get()
            if embedder_type == "simple":
                self.embedder = SimpleFaceEmbedder()
            
            if not self.embedder:
                messagebox.showerror("Error", "Failed to initialize face embedder")
                return
            
            # Start processing in separate thread
            self.process_btn.config(state='disabled')
            self.progress_var.set(0)
            self.results_text.delete(1.0, tk.END)
            
            processing_thread = threading.Thread(target=self._process_students_thread)
            processing_thread.daemon = True
            processing_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start processing: {str(e)}")
            self.process_btn.config(state='normal')
    
    def _process_students_thread(self):
        """Process students in separate thread"""
        try:
            total_students = len(self.students_data)
            successful_registrations = 0
            failed_registrations = 0
            
            self.processing_results = []
            
            for i, student in enumerate(self.students_data):
                try:
                    # Update progress
                    progress = (i / total_students) * 100
                    self.dialog.after(0, lambda p=progress: self.progress_var.set(p))
                    
                    # Process student
                    result = self._process_single_student(student)
                    self.processing_results.append(result)
                    
                    if result['success']:
                        successful_registrations += 1
                    else:
                        failed_registrations += 1
                    
                    # Update results display
                    result_text = f"Student {student['student_id']}: {'SUCCESS' if result['success'] else 'FAILED'}"
                    if not result['success']:
                        result_text += f" - {result['error']}"
                    result_text += "\n"
                    
                    self.dialog.after(0, lambda text=result_text: self.results_text.insert(tk.END, text))
                    
                except Exception as e:
                    error_result = {
                        'student_id': student['student_id'],
                        'success': False,
                        'error': str(e)
                    }
                    self.processing_results.append(error_result)
                    failed_registrations += 1
                    
                    error_text = f"Student {student['student_id']}: FAILED - {str(e)}\n"
                    self.dialog.after(0, lambda text=error_text: self.results_text.insert(tk.END, text))
            
            # Final update
            self.dialog.after(0, lambda: self.progress_var.set(100))
            
            summary = f"\n=== PROCESSING COMPLETE ===\n"
            summary += f"Total Students: {total_students}\n"
            summary += f"Successful: {successful_registrations}\n"
            summary += f"Failed: {failed_registrations}\n"
            summary += f"Success Rate: {(successful_registrations/total_students)*100:.1f}%\n"
            
            self.dialog.after(0, lambda text=summary: self.results_text.insert(tk.END, text))
            
            # Update students preview with quality scores
            self.dialog.after(0, self._update_students_preview_with_quality)
            
            # Re-enable process button
            self.dialog.after(0, lambda: self.process_btn.config(state='normal'))
            
            # Show completion message
            self.dialog.after(0, lambda: messagebox.showinfo("Processing Complete", 
                                                           f"Processed {total_students} students\n" 
                                                           f"Success: {successful_registrations}\n" 
                                                           f"Failed: {failed_registrations}"))
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.dialog.after(0, lambda: self.results_text.insert(tk.END, error_msg))
            self.dialog.after(0, lambda: self.process_btn.config(state='normal'))
    
    def _process_single_student(self, student: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single student registration"""
        try:
            student_id = student['student_id']
            
            # Check if student already exists
            if self.database_manager.get_student(student_id):
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Student already exists'
                }
            
            # Check if photo exists
            if student_id not in self.photo_files:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Photo not found'
                }
            
            # Load and process photo
            photo_path = self.photo_files[student_id]
            image = cv2.imread(photo_path)
            if image is None:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Invalid image file'
                }
            
            # Detect face in image
            faces = self.face_detector.detect_faces(image)
            if not faces:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'No face detected in photo'
                }
            
            # Use the first detected face
            face_box = faces[0]
            
            # Extract face
            face_image = self.face_detector.extract_face_region(image, face_box)
            if face_image is None:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Failed to extract face region'
                }
            
            # Generate embedding
            embedding = self.embedder.get_embedding(face_image)
            if embedding is None:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Failed to generate face embedding'
                }

            # Check for duplicate face
            duplicate_student_id = self._check_for_duplicate_face(embedding)
            if duplicate_student_id:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': f'Face already registered to student ID: {duplicate_student_id}'
                }
            
            # Check quality
            is_acceptable, reason, quality_metrics = self.quality_checker.check_new_encoding_quality(
                embedding, [], [], student_id
            )
            
            if not is_acceptable:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': f'Quality too low: {reason}'
                }
            
            # Add student to database
            if not self.database_manager.add_student(student):
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Failed to add student to database (already exists?)'
                }
            
            # Save face image
            face_data_dir = self.config.get('face_data_path', 'face_data')
            os.makedirs(face_data_dir, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            face_filename = f"{student_id}_{timestamp}.jpg"
            face_filepath = os.path.join(face_data_dir, face_filename)
            
            cv2.imwrite(face_filepath, face_image)
            
            # Add face encoding
            encoding_success = self.database_manager.add_face_encodings(
                student_id, [embedding], self.embedder_var.get(), [face_filepath]
            )
            
            if not encoding_success:
                # Rollback student addition if encoding fails
                self.database_manager.delete_student(student_id)
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Failed to add face encoding'
                }
            
            return {
                'student_id': student_id,
                'success': True,
                'quality_score': quality_metrics.get('quality_score', 0.0),
                'quality_msg': reason
            }
            
        except Exception as e:
            return {
                'student_id': student.get('student_id', 'Unknown'),
                'success': False,
                'error': str(e)
            }
    
    def _detect_face(self, image):
        """Detect face in image"""
        try:
            faces = self.face_detector.detect_faces(image)
            if faces:
                # Return the first detected face
                return faces[0]
            return None
        except Exception as e:
            print(f"Error detecting face: {e}")
            return None
    
    def _check_face_quality(self, face_image):
        """Check face image quality"""
        try:
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
            
            # Blur check
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
    
    def _update_students_preview_with_quality(self):
        """Update students preview with quality scores from processing results"""
        try:
            # Create quality lookup
            quality_lookup = {}
            for result in self.processing_results:
                if result['success']:
                    quality_lookup[result['student_id']] = f"{result['quality_score']:.2f}"
                else:
                    quality_lookup[result['student_id']] = "Failed"
            
            # Update tree items
            for item in self.students_tree.get_children():
                values = list(self.students_tree.item(item)['values'])
                student_id = values[0]
                
                if student_id in quality_lookup:
                    values[-1] = quality_lookup[student_id]  # Update quality column
                    self.students_tree.item(item, values=values)
            
        except Exception as e:
            print(f"Error updating preview with quality: {e}")
    
    def _close_dialog(self):
        """Close the dialog"""
        try:
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
