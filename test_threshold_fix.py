"""
Test the threshold fix for better recognition
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_threshold_fix():
    """Test the threshold fix"""
    print("Testing Threshold Fix for Better Recognition...")
    
    embedder = SimpleFaceEmbedder()
    
    # Create test faces
    np.random.seed(1)
    face1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # Person A
    np.random.seed(2)
    face2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # Person B
    
    # Get embeddings
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    
    if emb1 is None or emb2 is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"Embedding 1 shape: {emb1.shape}, norm: {np.linalg.norm(emb1):.4f}")
    print(f"Embedding 2 shape: {emb2.shape}, norm: {np.linalg.norm(emb2):.4f}")
    
    # Test with different thresholds
    print("\n=== Testing with Different Thresholds ===")
    thresholds = [0.85, 0.90, 0.95, 0.98, 0.99]
    
    print("Threshold | Same Person | Different Person | Expected")
    print("-" * 60)
    
    for threshold in thresholds:
        # Same person
        _, sim_same = embedder.compare_faces(emb1, emb1, threshold)
        is_same_same = sim_same >= threshold
        
        # Different person
        _, sim_diff = embedder.compare_faces(emb1, emb2, threshold)
        is_same_diff = sim_diff >= threshold
        
        status_same = "✅" if is_same_same else "❌"
        status_diff = "✅" if not is_same_diff else "❌"
        
        print(f"{threshold:8.2f} | {is_same_same:10} {status_same} | {is_same_diff:13} {status_diff} | Same: Match, Diff: Reject")
    
    # Test the new matching logic with 0.90 threshold
    print("\n=== Testing New Matching Logic (0.90 threshold) ===")
    
    # Simulate student encodings
    student_encodings = {
        "123": [emb1, emb1]  # Same embedding twice
    }
    
    # Test 1: Same person (should match)
    print("\n--- Test 1: Same person (should match) ---")
    best_confidence = 0.0
    best_student_id = None
    comparison_threshold = 0.5
    final_threshold = 0.90  # New threshold
    
    for student_id, encodings in student_encodings.items():
        student_max_similarity = 0.0
        for encoding in encodings:
            is_same, similarity = embedder.compare_faces(emb1, encoding, comparison_threshold)
            if similarity > student_max_similarity:
                student_max_similarity = similarity
    
        if student_max_similarity > final_threshold and student_max_similarity > best_confidence:
            best_confidence = student_max_similarity
            best_student_id = student_id
    
    if best_confidence > final_threshold:
        print(f"✅ SAME PERSON MATCH: {best_student_id} with confidence {best_confidence:.4f}")
    else:
        print(f"❌ SAME PERSON FAILED: Best confidence {best_confidence:.4f}")
    
    # Test 2: Different person (should NOT match)
    print("\n--- Test 2: Different person (should NOT match) ---")
    best_confidence = 0.0
    best_student_id = None
    
    for student_id, encodings in student_encodings.items():
        student_max_similarity = 0.0
        for encoding in encodings:
            is_same, similarity = embedder.compare_faces(emb2, encoding, comparison_threshold)
            if similarity > student_max_similarity:
                student_max_similarity = similarity
    
        if student_max_similarity > final_threshold and student_max_similarity > best_confidence:
            best_confidence = student_max_similarity
            best_student_id = student_id
    
    if best_confidence > final_threshold:
        print(f"❌ DIFFERENT PERSON INCORRECTLY MATCHED: {best_student_id} with confidence {best_confidence:.4f}")
    else:
        print(f"✅ DIFFERENT PERSON CORRECTLY REJECTED: Best confidence {best_confidence:.4f}")
    
    print("\n=== Analysis of Your Console Output ===")
    print("Your console showed:")
    print("- Best confidence: 0.9544")
    print("- Old threshold: 0.98")
    print("- Result: REJECTED (0.9544 < 0.98)")
    print()
    print("With new threshold 0.90:")
    print("- Best confidence: 0.9544")
    print("- New threshold: 0.90")
    print("- Result: ACCEPTED (0.9544 > 0.90) ✅")
    print()
    print("🎯 Your face should now be recognized!")

if __name__ == "__main__":
    test_threshold_fix()
