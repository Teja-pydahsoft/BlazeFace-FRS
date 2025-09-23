"""
Test the strict threshold to see if it properly rejects different people
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder
from app.core.database import DatabaseManager

def test_strict_threshold():
    """Test the strict threshold"""
    print("=== Testing Strict Threshold (0.99) ===")
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
    
    # Test the actual matching logic with 0.99 threshold
    print("\n=== Testing Matching Logic with 0.99 Threshold ===")
    
    def test_matching_logic(query_embedding, student_encodings, threshold=0.99):
        """Test the actual matching logic with strict threshold"""
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
        
        if best_confidence > 0.99:
            print(f"✅ MATCH FOUND: {best_student_id} with confidence {best_confidence:.4f}")
            return best_student_id, best_confidence
        else:
            print(f"❌ NO MATCH: Best confidence {best_confidence:.4f} below 0.99 threshold")
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
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    if match1 == "123" and match2 is None:
        print("✅ PERFECT: Same face matches, different face rejected")
        print("🎯 The 0.99 threshold is working correctly!")
    elif match1 == "123" and match2 == "123":
        print("❌ PROBLEM: Different faces are still being matched to same student")
        print("🔧 Need even stricter threshold or better embedder")
    elif match1 is None and match2 is None:
        print("❌ PROBLEM: Even same face is not being recognized")
        print("🔧 Threshold might be too strict")
    else:
        print(f"⚠️  MIXED: Face 1: {match1}, Face 2: {match2}")
    
    # Test with your actual face data
    print(f"\n=== Testing with Your Actual Face Data ===")
    print("Based on your console output:")
    print("- Your face confidence: 0.9544")
    print("- With 0.99 threshold: 0.9544 < 0.99 = REJECTED")
    print("- This means your face will show as 'Unknown'")
    print()
    print("🔧 SOLUTION: We need to find the right balance:")
    print("1. High enough to reject different people")
    print("2. Low enough to accept the same person")
    print()
    print("Let's test with 0.95 threshold...")
    
    # Test with 0.95 threshold
    print(f"\n--- Testing with 0.95 Threshold ---")
    match1_95, conf1_95 = test_matching_logic(emb1, student_encodings, 0.95)
    match2_95, conf2_95 = test_matching_logic(emb2, student_encodings, 0.95)
    
    print(f"Face 1 with 0.95: {match1_95} (confidence: {conf1_95:.4f})")
    print(f"Face 2 with 0.95: {match2_95} (confidence: {conf2_95:.4f})")
    
    if match1_95 == "123" and match2_95 is None:
        print("✅ 0.95 threshold works: Same face matches, different face rejected")
        print("🎯 RECOMMENDATION: Use 0.95 threshold")
    elif match1_95 == "123" and match2_95 == "123":
        print("❌ 0.95 threshold still allows false matches")
        print("🔧 Need to stick with 0.99 threshold")
    else:
        print("⚠️  Mixed results with 0.95 threshold")

if __name__ == "__main__":
    test_strict_threshold()
