"""
Debug the encoding comparison logic
"""

import sys
import os
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from app.core.simple_face_embedder import SimpleFaceEmbedder

def debug_encoding_comparison():
    """Debug the encoding comparison logic"""
    try:
        print("Debugging Encoding Comparison Logic...")
        
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
        
        # Show encoding details
        for student_id, encodings_list in student_encodings.items():
            print(f"\nStudent {student_id}: {len(encodings_list)} encodings")
            for i, encoding in enumerate(encodings_list):
                print(f"  Encoding {i}: shape={encoding.shape}, norm={np.linalg.norm(encoding):.4f}")
                print(f"  Encoding {i}: sample values={encoding[:5]}")
        
        # Test 1: Same encoding vs itself
        if student_encodings:
            student_id = list(student_encodings.keys())[0]
            encoding1 = student_encodings[student_id][0]
            
            print(f"\n=== Test 1: Same encoding vs itself (Student: {student_id}) ===")
            is_same, similarity = embedder.compare_faces(encoding1, encoding1, 0.9)
            print(f"Result: is_same={is_same}, similarity={similarity:.4f}")
            print(f"Expected: is_same=True, similarity=1.0000")
        
        # Test 2: Different encodings from same student
        if student_encodings and len(student_encodings[list(student_encodings.keys())[0]]) > 1:
            student_id = list(student_encodings.keys())[0]
            encoding1 = student_encodings[student_id][0]
            encoding2 = student_encodings[student_id][1]
            
            print(f"\n=== Test 2: Different encodings from same student (Student: {student_id}) ===")
            is_same, similarity = embedder.compare_faces(encoding1, encoding2, 0.9)
            print(f"Result: is_same={is_same}, similarity={similarity:.4f}")
            print(f"Expected: is_same=False (different encodings should not match at 0.9 threshold)")
        
        # Test 3: Create a completely different encoding
        print(f"\n=== Test 3: Same encoding vs random encoding ===")
        if student_encodings:
            student_id = list(student_encodings.keys())[0]
            encoding1 = student_encodings[student_id][0]
            # Create a random encoding
            random_encoding = np.random.rand(128).astype(np.float32)
            # Normalize it
            random_encoding = random_encoding / np.linalg.norm(random_encoding)
            
            is_same, similarity = embedder.compare_faces(encoding1, random_encoding, 0.9)
            print(f"Result: is_same={is_same}, similarity={similarity:.4f}")
            print(f"Expected: is_same=False (random encoding should not match)")
        
        # Test 4: Simulate the _find_best_match logic
        print(f"\n=== Test 4: Simulating _find_best_match logic ===")
        if student_encodings:
            student_id = list(student_encodings.keys())[0]
            query_encoding = student_encodings[student_id][0]  # Use first encoding as query
            
            print(f"Query encoding: shape={query_encoding.shape}, norm={np.linalg.norm(query_encoding):.4f}")
            
            best_confidence = 0.0
            best_student_id = None
            
            for student_id, encodings_list in student_encodings.items():
                print(f"Checking student {student_id} with {len(encodings_list)} encodings")
                student_max_similarity = 0.0
                
                for i, encoding in enumerate(encodings_list):
                    is_same, similarity = embedder.compare_faces(query_encoding, encoding, 0.9)
                    print(f"  Encoding {i}: is_same={is_same}, similarity={similarity:.4f}")
                    if is_same and similarity > student_max_similarity:
                        student_max_similarity = similarity
                
                print(f"  Student {student_id} max similarity: {student_max_similarity:.4f}")
                
                if student_max_similarity > 0.9 and student_max_similarity > best_confidence:
                    best_confidence = student_max_similarity
                    best_student_id = student_id
                    print(f"  -> New best match: {student_id} with {student_max_similarity:.4f}")
            
            print(f"Final result: best_confidence={best_confidence:.4f}, best_student_id={best_student_id}")
            
            if best_confidence > 0.9:
                print(f"✓ MATCH FOUND: {best_student_id} with confidence {best_confidence:.4f}")
            else:
                print(f"✗ NO MATCH: Best confidence {best_confidence:.4f} below 0.9 threshold")
        
    except Exception as e:
        print(f"Error in encoding debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_encoding_comparison()
