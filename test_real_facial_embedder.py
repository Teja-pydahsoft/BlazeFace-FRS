"""
Test the real facial embedder to see if it can distinguish between different faces
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.real_facial_embedder import RealFacialEmbedder

def create_realistic_face1():
    """Create a realistic face image 1"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Face shape
    cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (200, 180, 160), -1)
    
    # Eyes
    cv2.circle(img, (80, 80), 8, (0, 0, 0), -1)
    cv2.circle(img, (120, 80), 8, (0, 0, 0), -1)
    
    # Nose
    cv2.ellipse(img, (100, 110), (5, 15), 0, 0, 360, (150, 130, 110), -1)
    
    # Mouth
    cv2.ellipse(img, (100, 130), (15, 8), 0, 0, 180, (0, 0, 0), -1)
    
    return img

def create_realistic_face2():
    """Create a realistic face image 2 (different person)"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Face shape (wider)
    cv2.ellipse(img, (100, 100), (70, 85), 0, 0, 360, (190, 170, 150), -1)
    
    # Eyes (different positions)
    cv2.circle(img, (75, 85), 10, (0, 0, 0), -1)
    cv2.circle(img, (125, 85), 10, (0, 0, 0), -1)
    
    # Nose (different shape)
    cv2.ellipse(img, (100, 115), (8, 20), 0, 0, 360, (140, 120, 100), -1)
    
    # Mouth (different shape)
    cv2.ellipse(img, (100, 135), (20, 10), 0, 0, 180, (0, 0, 0), -1)
    
    return img

def test_real_facial_embedder():
    """Test the real facial embedder"""
    print("=== Testing Real Facial Embedder ===")
    print()
    
    embedder = RealFacialEmbedder()
    
    # Create two different realistic faces
    face1 = create_realistic_face1()
    face2 = create_realistic_face2()
    
    print("Created two different realistic faces")
    
    # Get embeddings
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    
    if emb1 is None or emb2 is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"Face 1 embedding shape: {emb1.shape}, norm: {np.linalg.norm(emb1):.4f}")
    print(f"Face 2 embedding shape: {emb2.shape}, norm: {np.linalg.norm(emb2):.4f}")
    
    # Test consistency (same face should match)
    print("\n=== Testing Consistency ===")
    emb1_again = embedder.get_embedding(face1)
    is_same_1, sim_1 = embedder.compare_faces(emb1, emb1_again, 0.8)
    print(f"Same face (1 vs 1): {is_same_1} (similarity: {sim_1:.4f})")
    
    # Test discrimination (different faces should NOT match)
    print("\n=== Testing Discrimination ===")
    is_same_diff, sim_diff = embedder.compare_faces(emb1, emb2, 0.8)
    print(f"Different faces (1 vs 2): {is_same_diff} (similarity: {sim_diff:.4f})")
    
    # Test with different thresholds
    print("\n=== Testing with Different Thresholds ===")
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.98]
    
    print("Threshold | Same Person | Different Person | Expected")
    print("-" * 60)
    
    for threshold in thresholds:
        # Same person
        _, sim_same = embedder.compare_faces(emb1, emb1_again, threshold)
        is_same_same = sim_same >= threshold
        
        # Different person
        _, sim_diff = embedder.compare_faces(emb1, emb2, threshold)
        is_same_diff = sim_diff >= threshold
        
        status_same = "✅" if is_same_same else "❌"
        status_diff = "✅" if not is_same_diff else "❌"
        
        print(f"{threshold:8.2f} | {is_same_same:10} {status_same} | {is_same_diff:13} {status_diff} | Same: Match, Diff: Reject")
    
    # Test the actual matching logic
    print("\n=== Testing Matching Logic ===")
    
    # Simulate student encodings
    student_encodings = {
        "123": [emb1, emb1]  # Same embedding twice
    }
    
    def test_matching_logic(query_embedding, student_encodings, threshold=0.8):
        """Test the actual matching logic"""
        best_confidence = 0.0
        best_student_id = None
        
        for student_id, encodings in student_encodings.items():
            student_max_similarity = 0.0
            
            for encoding in encodings:
                is_same, similarity = embedder.compare_faces(query_embedding, encoding, 0.5)
                if similarity > student_max_similarity:
                    student_max_similarity = similarity
            
            if student_max_similarity > threshold and student_max_similarity > best_confidence:
                best_confidence = student_max_similarity
                best_student_id = student_id
        
        return best_student_id, best_confidence
    
    # Test with face 1 (should match)
    print("\n--- Testing Face 1 (should match) ---")
    match1, conf1 = test_matching_logic(emb1, student_encodings, 0.8)
    print(f"Face 1: {match1} (confidence: {conf1:.4f})")
    
    # Test with face 2 (should NOT match)
    print("\n--- Testing Face 2 (should NOT match) ---")
    match2, conf2 = test_matching_logic(emb2, student_encodings, 0.8)
    print(f"Face 2: {match2} (confidence: {conf2:.4f})")
    
    # Analysis
    print("\n=== ANALYSIS ===")
    if match1 == "123" and match2 is None:
        print("✅ PERFECT: Same face matches, different face rejected")
    elif match1 == "123" and match2 == "123":
        print("❌ PROBLEM: Different faces are being matched to same student")
    elif match1 is None and match2 is None:
        print("❌ PROBLEM: Even same face is not being recognized")
    else:
        print(f"⚠️  MIXED: Face 1: {match1}, Face 2: {match2}")
    
    print(f"\nSimilarity scores:")
    print(f"  Same face: {sim_1:.4f}")
    print(f"  Different faces: {sim_diff:.4f}")
    print(f"  Difference: {sim_1 - sim_diff:.4f}")
    
    if sim_1 - sim_diff > 0.1:
        print("✅ Good discrimination (difference > 0.1)")
    else:
        print("❌ Poor discrimination (difference <= 0.1)")

if __name__ == "__main__":
    test_real_facial_embedder()
