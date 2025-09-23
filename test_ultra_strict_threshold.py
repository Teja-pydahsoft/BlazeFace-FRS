"""
Test the ultra-strict threshold (0.995) to see if it properly rejects different people
"""

import sys
import os
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_ultra_strict_threshold():
    """Test the ultra-strict threshold"""
    print("=== Testing Ultra-Strict Threshold (0.995) ===")
    print()
    
    embedder = SimpleFaceEmbedder()
    
    # Create test data
    np.random.seed(1)
    face1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    np.random.seed(2)
    face2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    
    # Simulate student encodings
    student_encodings = {
        "123": [emb1, emb1]  # Same embedding twice
    }
    
    def test_matching_logic(query_embedding, student_encodings, threshold=0.995):
        """Test the actual matching logic with ultra-strict threshold"""
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
    
    # Test with same face (should match)
    print("--- Testing Same Face (should match) ---")
    match1, conf1 = test_matching_logic(emb1, student_encodings, 0.995)
    print(f"Same face: {match1} (confidence: {conf1:.4f})")
    
    # Test with different face (should NOT match)
    print("\n--- Testing Different Face (should NOT match) ---")
    match2, conf2 = test_matching_logic(emb2, student_encodings, 0.995)
    print(f"Different face: {match2} (confidence: {conf2:.4f})")
    
    # Test consistency
    print("\n--- Testing Consistency ---")
    emb1_again = embedder.get_embedding(face1)
    match1_again, conf1_again = test_matching_logic(emb1_again, student_encodings, 0.995)
    print(f"Same face again: {match1_again} (confidence: {conf1_again:.4f})")
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    if match1 == "123" and match2 is None:
        print("✅ PERFECT: Same face matches, different face rejected")
        print("🎯 The 0.995 threshold is working correctly!")
    elif match1 == "123" and match2 == "123":
        print("❌ PROBLEM: Different faces are still being matched")
        print("🔧 Need even stricter threshold")
    elif match1 is None and match2 is None:
        print("❌ PROBLEM: Even same face is not being recognized")
        print("🔧 Threshold is too strict")
    else:
        print(f"⚠️  MIXED: Same: {match1}, Different: {match2}")
    
    # Test with your actual face data
    print(f"\n=== Testing with Your Actual Face Data ===")
    print("Based on your console output:")
    print("- Your face confidence: 0.9544")
    print("- With 0.995 threshold: 0.9544 < 0.995 = REJECTED")
    print("- This means your face will show as 'Unknown'")
    print()
    print("🔧 SOLUTION: We need to find the right balance:")
    print("1. High enough to reject different people (0.9781)")
    print("2. Low enough to accept the same person (1.0000)")
    print()
    print("Let's test with 0.98 threshold...")
    
    # Test with 0.98 threshold
    print(f"\n--- Testing with 0.98 Threshold ---")
    match1_98, conf1_98 = test_matching_logic(emb1, student_encodings, 0.98)
    match2_98, conf2_98 = test_matching_logic(emb2, student_encodings, 0.98)
    
    print(f"Same face with 0.98: {match1_98} (confidence: {conf1_98:.4f})")
    print(f"Different face with 0.98: {match2_98} (confidence: {conf2_98:.4f})")
    
    if match1_98 == "123" and match2_98 is None:
        print("✅ 0.98 threshold works: Same face matches, different face rejected")
        print("🎯 RECOMMENDATION: Use 0.98 threshold")
    elif match1_98 == "123" and match2_98 == "123":
        print("❌ 0.98 threshold still allows false matches")
        print("🔧 Need to stick with 0.995 threshold")
    else:
        print("⚠️  Mixed results with 0.98 threshold")
    
    # Test with 0.99 threshold
    print(f"\n--- Testing with 0.99 Threshold ---")
    match1_99, conf1_99 = test_matching_logic(emb1, student_encodings, 0.99)
    match2_99, conf2_99 = test_matching_logic(emb2, student_encodings, 0.99)
    
    print(f"Same face with 0.99: {match1_99} (confidence: {conf1_99:.4f})")
    print(f"Different face with 0.99: {match2_99} (confidence: {conf2_99:.4f})")
    
    if match1_99 == "123" and match2_99 is None:
        print("✅ 0.99 threshold works: Same face matches, different face rejected")
        print("🎯 RECOMMENDATION: Use 0.99 threshold")
    elif match1_99 == "123" and match2_99 == "123":
        print("❌ 0.99 threshold still allows false matches")
        print("🔧 Need to stick with 0.995 threshold")
    else:
        print("⚠️  Mixed results with 0.99 threshold")

if __name__ == "__main__":
    test_ultra_strict_threshold()
