"""
Manual Photo Upload Test for Face Recognition System
Test if registered students can be recognized from uploaded photos
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from app.core.simple_face_embedder import SimpleFaceEmbedder
from app.core.blazeface_detector import BlazeFaceDetector

class ManualPhotoTest:
    def __init__(self):
        """Initialize the manual photo test application"""
        self.root = tk.Tk()
        self.root.title("Manual Photo Upload Test - BlazeFace-FRS")
        self.root.geometry("800x600")
        
        # Initialize components
        self.db_manager = DatabaseManager("database/blazeface_frs.db")
        self.embedder = SimpleFaceEmbedder()
        self.detector = BlazeFaceDetector(min_detection_confidence=0.5)
        
        # Load student data
        self.student_encodings = {}
        self.student_names = {}
        self._load_student_data()
        
        # Setup UI
        self._setup_ui()
        
    def _load_student_data(self):
        """Load student data and encodings"""
        try:
            # Get face encodings
            encodings = self.db_manager.get_face_encodings()
            print(f"Loaded {len(encodings)} face encodings from database")
            
            for student_id, encoding, encoding_type in encodings:
                if student_id not in self.student_encodings:
                    self.student_encodings[student_id] = []
                    # Get student name
                    student_info = self.db_manager.get_student(student_id)
                    if student_info:
                        self.student_names[student_id] = student_info['name']
                    else:
                        self.student_names[student_id] = f"Student {student_id}"
                self.student_encodings[student_id].append(encoding)
            
            print(f"Loaded {len(self.student_encodings)} students:")
            for student_id, name in self.student_names.items():
                print(f"  - {student_id}: {name} ({len(self.student_encodings[student_id])} encodings)")
                
        except Exception as e:
            print(f"Error loading student data: {e}")
            self.student_encodings = {}
            self.student_names = {}
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Manual Photo Upload Test", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Upload section
        upload_frame = ttk.LabelFrame(left_frame, text="Photo Upload", padding="10")
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(upload_frame, text="Select Photo", 
                  command=self._select_photo).pack(fill=tk.X, pady=5)
        
        ttk.Button(upload_frame, text="Test Recognition", 
                  command=self._test_recognition).pack(fill=tk.X, pady=5)
        
        # Settings section
        settings_frame = ttk.LabelFrame(left_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Recognition threshold
        ttk.Label(settings_frame, text="Recognition Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.9)
        threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                  variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Face detection confidence
        ttk.Label(settings_frame, text="Min Face Confidence:").pack(anchor=tk.W)
        self.face_confidence_var = tk.DoubleVar(value=0.7)
        face_confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                        variable=self.face_confidence_var, orient=tk.HORIZONTAL)
        face_confidence_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Student info section
        info_frame = ttk.LabelFrame(left_frame, text="Student Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(info_frame, height=10, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Image display
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display
        image_frame = ttk.LabelFrame(right_frame, text="Photo Preview", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(image_frame, text="No photo selected", 
                                   font=("Arial", 12))
        self.image_label.pack(expand=True)
        
        # Results section
        results_frame = ttk.LabelFrame(right_frame, text="Recognition Results", padding="10")
        results_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.results_text = tk.Text(results_frame, height=8, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize
        self.current_image = None
        self.current_faces = []
        self._update_student_info()
    
    def _update_student_info(self):
        """Update student information display"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"Registered Students: {len(self.student_encodings)}\n\n")
        
        for student_id, name in self.student_names.items():
            encodings_count = len(self.student_encodings[student_id])
            self.info_text.insert(tk.END, f"• {student_id}: {name}\n")
            self.info_text.insert(tk.END, f"  Encodings: {encodings_count}\n\n")
    
    def _select_photo(self):
        """Select a photo file"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load image
                self.current_image = cv2.imread(filename)
                if self.current_image is None:
                    messagebox.showerror("Error", "Could not load image file")
                    return
                
                # Display image
                self._display_image(self.current_image)
                
                # Detect faces
                self._detect_faces()
                
                self._log_result(f"Photo loaded: {os.path.basename(filename)}")
                self._log_result(f"Image size: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def _display_image(self, image):
        """Display image in the preview area"""
        try:
            # Resize image for display
            height, width = image.shape[:2]
            max_width = 400
            max_height = 300
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.config(text=f"Error displaying image: {str(e)}")
    
    def _detect_faces(self):
        """Detect faces in the current image"""
        if self.current_image is None:
            return
        
        try:
            # Detect faces
            faces = self.detector.detect_faces(self.current_image)
            self.current_faces = faces
            
            self._log_result(f"Detected {len(faces)} faces")
            
            # Draw faces on image
            image_with_faces = self.current_image.copy()
            for i, face in enumerate(faces):
                x, y, w, h, confidence = face
                cv2.rectangle(image_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_with_faces, f"Face {i}: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display image with faces
            self._display_image(image_with_faces)
            
        except Exception as e:
            self._log_result(f"Error detecting faces: {str(e)}")
    
    def _test_recognition(self):
        """Test recognition on detected faces"""
        if not self.current_faces:
            messagebox.showwarning("Warning", "No faces detected. Please select a photo with faces.")
            return
        
        self._log_result("\n" + "="*50)
        self._log_result("STARTING RECOGNITION TEST")
        self._log_result("="*50)
        
        threshold = self.threshold_var.get()
        min_face_confidence = self.face_confidence_var.get()
        
        self._log_result(f"Using recognition threshold: {threshold:.2f}")
        self._log_result(f"Using min face confidence: {min_face_confidence:.2f}")
        self._log_result(f"Comparing against {len(self.student_encodings)} students")
        
        for i, face in enumerate(self.current_faces):
            x, y, w, h, confidence = face
            
            self._log_result(f"\n--- Face {i} ---")
            self._log_result(f"Detection confidence: {confidence:.2f}")
            
            # Check face confidence
            if confidence < min_face_confidence:
                self._log_result(f"❌ SKIPPED: Low detection confidence {confidence:.2f} < {min_face_confidence:.2f}")
                continue
            
            # Extract face region
            face_region = self.detector.extract_face_region(self.current_image, (x, y, w, h))
            if face_region is None:
                self._log_result(f"❌ SKIPPED: Could not extract face region")
                continue
            
            # Get embedding
            embedding = self.embedder.get_embedding(face_region)
            if embedding is None:
                self._log_result(f"❌ SKIPPED: Could not generate embedding")
                continue
            
            self._log_result(f"Embedding shape: {embedding.shape}")
            self._log_result(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
            
            # Find best match
            best_match = self._find_best_match(embedding, threshold)
            
            if best_match:
                student_id, match_confidence = best_match
                student_name = self.student_names.get(student_id, f"Student {student_id}")
                self._log_result(f"✅ MATCH FOUND: {student_name} (ID: {student_id})")
                self._log_result(f"   Match confidence: {match_confidence:.4f}")
            else:
                self._log_result(f"❌ NO MATCH: Unknown person")
        
        self._log_result("\n" + "="*50)
        self._log_result("RECOGNITION TEST COMPLETED")
        self._log_result("="*50)
    
    def _find_best_match(self, query_embedding, threshold):
        """Find best matching student for the given embedding"""
        try:
            best_confidence = 0.0
            best_student_id = None
            
            # Use a lower threshold for comparison to get all similarities
            comparison_threshold = 0.5  # Lower threshold to get all similarities
            final_threshold = max(threshold, 0.98)  # Optimal threshold to prevent false matches
            
            self._log_result(f"Using comparison threshold: {comparison_threshold:.2f}")
            self._log_result(f"Using final threshold: {final_threshold:.2f}")
            
            for student_id, encodings in self.student_encodings.items():
                student_max_similarity = 0.0
                student_similarities = []
                
                for i, encoding in enumerate(encodings):
                    # Use lower threshold to get actual similarity scores
                    is_same, similarity = self.embedder.compare_faces(query_embedding, encoding, comparison_threshold)
                    student_similarities.append(similarity)
                    
                    # Track the highest similarity for this student
                    if similarity > student_max_similarity:
                        student_max_similarity = similarity
                
                # Log similarities for debugging
                self._log_result(f"Student {student_id} similarities: {[f'{s:.4f}' for s in student_similarities[:3]]}... (max: {student_max_similarity:.4f})")
                
                # Only consider as match if similarity is above final threshold
                if student_max_similarity > final_threshold and student_max_similarity > best_confidence:
                    best_confidence = student_max_similarity
                    best_student_id = student_id
                    self._log_result(f"  -> New best match: {student_id} with {student_max_similarity:.4f}")
            
            # Only return match if confidence is above final threshold
            if best_confidence > final_threshold:
                self._log_result(f"✅ FINAL MATCH: {best_student_id} with confidence {best_confidence:.4f}")
                return best_student_id, best_confidence
            else:
                self._log_result(f"❌ NO MATCH: Best confidence {best_confidence:.4f} below {final_threshold:.2f} threshold")
                return None
            
        except Exception as e:
            self._log_result(f"Error in face matching: {str(e)}")
            return None
    
    def _log_result(self, message):
        """Log a message to the results text area"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        print(message)  # Also print to console
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    try:
        print("Starting Manual Photo Upload Test...")
        app = ManualPhotoTest()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
