"""
Debug the attendance system to see what's actually happening
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder
from app.core.database import DatabaseManager

def debug_attendance_system():
    """Debug the attendance system"""
    print("=== DEBUGGING ATTENDANCE SYSTEM ===")
    print()
    
    # Initialize components
    embedder = SimpleFaceEmbedder()
    db_manager = DatabaseManager("database/blazeface_frs.db")
    
    # Get stored encodings
    encodings = db_manager.get_face_encodings()
    print(f"Loaded {len(encodings)} face encodings from database")
    
    if not encodings:
        print("❌ No encodings found in database!")
        return
    
    # Create student encodings dictionary
    student_encodings = {}
    for student_id, encoding, encoding_type in encodings:
        if student_id not in student_encodings:
            student_encodings[student_id] = []
        student_encodings[student_id].append(encoding)
    
    print(f"Student encodings: {list(student_encodings.keys())}")
    
    # Test with different faces
    print("\n=== Testing with Different Faces ===")
    
    # Create two very different faces
    np.random.seed(1)
    face1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    np.random.seed(2)
    face2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Get embeddings
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    
    if emb1 is None or emb2 is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"Face 1 embedding norm: {np.linalg.norm(emb1):.4f}")
    print(f"Face 2 embedding norm: {np.linalg.norm(emb2):.4f}")
    
    # Test the actual matching logic used in attendance system
    print("\n=== Testing Attendance Matching Logic ===")
    
    def test_matching_logic(query_embedding, student_encodings, threshold=0.8):
        """Test the actual matching logic from attendance system"""
        best_confidence = 0.0
        best_student_id = None
        all_similarities = []
        
        for student_id, encodings in student_encodings.items():
            print(f"Checking student {student_id} with {len(encodings)} encodings")
            student_max_similarity = 0.0
            
            for i, encoding in enumerate(encodings):
                # Use the same logic as attendance system
                is_same, similarity = embedder.compare_faces(query_embedding, encoding, 0.5)
                all_similarities.append((student_id, i, similarity))
                
                if similarity > student_max_similarity:
                    student_max_similarity = similarity
                print(f"  Encoding {i}: similarity={similarity:.4f}")
            
            print(f"  Student {student_id} max similarity: {student_max_similarity:.4f}")
            
            if student_max_similarity > threshold and student_max_similarity > best_confidence:
                best_confidence = student_max_similarity
                best_student_id = student_id
                print(f"  -> New best match: {student_id} with {student_max_similarity:.4f}")
        
        print("All similarities:")
        for student_id, enc_idx, sim in sorted(all_similarities, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  {student_id}[{enc_idx}]: {sim:.4f}")
        
        print(f"Final result: best_confidence={best_confidence:.4f}, best_student_id={best_student_id}")
        
        if best_confidence > 0.90:
            print(f"✓ MATCH FOUND: {best_student_id} with confidence {best_confidence:.4f}")
            return best_student_id, best_confidence
        else:
            print(f"✗ NO MATCH: Best confidence {best_confidence:.4f} below 0.90 threshold")
            return None, 0.0
    
    # Test with face 1
    print("\n--- Testing Face 1 ---")
    match1, conf1 = test_matching_logic(emb1, student_encodings)
    
    # Test with face 2
    print("\n--- Testing Face 2 ---")
    match2, conf2 = test_matching_logic(emb2, student_encodings)
    
    # Test consistency
    print("\n--- Testing Consistency ---")
    emb1_again = embedder.get_embedding(face1)
    match1_again, conf1_again = test_matching_logic(emb1_again, student_encodings)
    
    print(f"\n=== RESULTS ===")
    print(f"Face 1: {match1} (confidence: {conf1:.4f})")
    print(f"Face 2: {match2} (confidence: {conf2:.4f})")
    print(f"Face 1 again: {match1_again} (confidence: {conf1_again:.4f})")
    
    # Check if different faces are getting matched to same student
    if match1 == match2 and match1 is not None:
        print("❌ PROBLEM: Different faces are being matched to the same student!")
        print("This explains why you see different people recognized as 'Teja'")
    else:
        print("✅ Different faces are correctly identified as different")
    
    # Check consistency
    if match1 == match1_again:
        print("✅ Same face is consistently recognized")
    else:
        print("❌ PROBLEM: Same face is not consistently recognized")

if __name__ == "__main__":
    debug_attendance_system()
