"""
Test the encoding comparison fix
"""

import sys
import os
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_encoding_comparison():
    """Test the encoding comparison logic"""
    try:
        print("Testing Encoding Comparison Logic...")
        
        # Initialize components
        db_manager = DatabaseManager("database/blazeface_frs.db")
        embedder = SimpleFaceEmbedder()
        
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
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        
        # Test 1: Same encoding vs itself
        if student_encodings:
            student_id = list(student_encodings.keys())[0]
            encoding1 = student_encodings[student_id][0]
            encoding2 = student_encodings[student_id][0]  # Same encoding
            
            print(f"\nTest 1: Same encoding vs itself (Student: {student_id})")
            for threshold in thresholds:
                is_same, similarity = embedder.compare_faces(encoding1, encoding2, threshold)
                print(f"  Threshold {threshold:.2f}: is_same={is_same}, similarity={similarity:.4f}")
        
        # Test 2: Different encodings from same student
        if student_encodings and len(student_encodings[list(student_encodings.keys())[0]]) > 1:
            student_id = list(student_encodings.keys())[0]
            encoding1 = student_encodings[student_id][0]
            encoding2 = student_encodings[student_id][1]  # Different encoding from same student
            
            print(f"\nTest 2: Different encodings from same student (Student: {student_id})")
            for threshold in thresholds:
                is_same, similarity = embedder.compare_faces(encoding1, encoding2, threshold)
                print(f"  Threshold {threshold:.2f}: is_same={is_same}, similarity={similarity:.4f}")
        
        # Test 3: Create a completely different encoding
        print(f"\nTest 3: Same encoding vs random encoding")
        if student_encodings:
            student_id = list(student_encodings.keys())[0]
            encoding1 = student_encodings[student_id][0]
            # Create a random encoding
            random_encoding = np.random.rand(128).astype(np.float32)
            # Normalize it
            random_encoding = random_encoding / np.linalg.norm(random_encoding)
            
            for threshold in thresholds:
                is_same, similarity = embedder.compare_faces(encoding1, random_encoding, threshold)
                print(f"  Threshold {threshold:.2f}: is_same={is_same}, similarity={similarity:.4f}")
        
        print("\nExpected results:")
        print("- Same encoding vs itself: Should match at all thresholds")
        print("- Different encodings from same student: May or may not match depending on quality")
        print("- Same encoding vs random: Should NOT match at high thresholds (0.8+)")
        
    except Exception as e:
        print(f"Error in encoding comparison test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_encoding_comparison()
