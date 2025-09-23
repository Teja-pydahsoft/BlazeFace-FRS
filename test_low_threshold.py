"""
Test face recognition with lower threshold
"""

import sys
import os
import cv2
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from app.core.simple_face_embedder import SimpleFaceEmbedder
from app.core.blazeface_detector import BlazeFaceDetector

def test_low_threshold():
    """Test face recognition with lower threshold"""
    try:
        print("Testing Face Recognition with Low Threshold...")
        
        # Initialize components
        db_manager = DatabaseManager("database/blazeface_frs.db")
        embedder = SimpleFaceEmbedder()
        detector = BlazeFaceDetector(min_detection_confidence=0.5)
        
        # Get stored encodings
        encodings = db_manager.get_face_encodings()
        print(f"Loaded {len(encodings)} face encodings from database")
        
        if not encodings:
            print("No face encodings found in database!")
            return
        
        # Create student encodings dictionary
        student_encodings = {}
        for student_id, encoding, encoding_type in encodings:
            if student_id not in student_encodings:
                student_encodings[student_id] = []
            student_encodings[student_id].append(encoding)
        
        print(f"Student encodings: {list(student_encodings.keys())}")
        
        # Test with different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera opened successfully. Press 'q' to quit, 's' to test recognition")
        
        frame_count = 0
        while frame_count < 200:  # Test for 200 frames
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            if faces:
                print(f"Frame {frame_count}: Detected {len(faces)} faces")
                
                # Process each face
                for i, face_box in enumerate(faces):
                    x, y, w, h, confidence = face_box
                    print(f"  Face {i}: Confidence={confidence:.2f}")
                    
                    # Extract face region
                    face_region = detector.extract_face_region(frame, (x, y, w, h))
                    if face_region is not None:
                        # Get embedding
                        embedding = embedder.get_embedding(face_region)
                        if embedding is not None:
                            print(f"    Got embedding with shape {embedding.shape}")
                            
                            # Test with different thresholds
                            print("    Testing with different thresholds:")
                            for threshold in thresholds:
                                best_confidence = 0.0
                                best_student_id = None
                                
                                for student_id, encodings_list in student_encodings.items():
                                    for stored_encoding in encodings_list:
                                        is_same, similarity = embedder.compare_faces(embedding, stored_encoding, threshold)
                                        if is_same and similarity > best_confidence:
                                            best_confidence = similarity
                                            best_student_id = student_id
                                
                                if best_student_id:
                                    print(f"      Threshold {threshold:.1f}: MATCH {best_student_id} (confidence: {best_confidence:.4f})")
                                else:
                                    print(f"      Threshold {threshold:.1f}: No match")
                            
                            # Only test once per frame
                            break
                        else:
                            print(f"    Failed to get embedding")
                    else:
                        print(f"    Failed to extract face region")
            else:
                if frame_count % 50 == 0:  # Print every 50th frame
                    print(f"Frame {frame_count}: No faces detected")
            
            # Draw faces
            frame_with_faces = detector.draw_faces(frame, faces)
            
            # Display frame
            cv2.imshow('Low Threshold Test', frame_with_faces)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and faces:
                # Test recognition with current face
                face_box = faces[0]
                x, y, w, h, confidence = face_box
                face_region = detector.extract_face_region(frame, (x, y, w, h))
                if face_region is not None:
                    embedding = embedder.get_embedding(face_region)
                    if embedding is not None:
                        print(f"\n=== RECOGNITION TEST ===")
                        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                            best_confidence = 0.0
                            best_student_id = None
                            
                            for student_id, encodings_list in student_encodings.items():
                                for stored_encoding in encodings_list:
                                    is_same, similarity = embedder.compare_faces(embedding, stored_encoding, threshold)
                                    if is_same and similarity > best_confidence:
                                        best_confidence = similarity
                                        best_student_id = student_id
                            
                            if best_student_id:
                                print(f"Threshold {threshold:.1f}: MATCH {best_student_id} (confidence: {best_confidence:.4f})")
                            else:
                                print(f"Threshold {threshold:.1f}: No match")
                        print("=== END TEST ===\n")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        embedder.release()
        
        print("Low threshold test completed")
        
    except Exception as e:
        print(f"Error in low threshold test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_low_threshold()
