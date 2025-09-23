"""
Test script to verify the discrimination fix works correctly
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_discrimination_fix():
    """Test that different faces are properly discriminated"""
    print("Testing Discrimination Fix...")
    
    embedder = SimpleFaceEmbedder()
    
    # Create two very different dummy faces
    np.random.seed(1)
    face1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    np.random.seed(2)
    face2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Get embeddings
    embedding1 = embedder.get_embedding(face1)
    embedding2 = embedder.get_embedding(face2)
    
    if embedding1 is None or embedding2 is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"Embedding 1 norm: {np.linalg.norm(embedding1):.4f}")
    print(f"Embedding 2 norm: {np.linalg.norm(embedding2):.4f}")
    
    # Test with different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    
    print("\nTesting face comparison with different thresholds:")
    print("Threshold | is_same | similarity | Expected")
    print("-" * 50)
    
    for threshold in thresholds:
        is_same, similarity = embedder.compare_faces(embedding1, embedding2, threshold)
        expected = "Different" if threshold >= 0.8 else "Unknown"
        status = "✅" if (not is_same and threshold >= 0.8) or (is_same and threshold < 0.8) else "❌"
        print(f"{threshold:8.2f} | {str(is_same):7} | {similarity:10.4f} | {expected:8} {status}")
    
    # Test the new matching logic
    print("\nTesting new matching logic:")
    
    # Simulate student encodings (same person - should match)
    student_encodings = {
        "student1": [embedding1, embedding1]  # Same embedding twice
    }
    
    # Test 1: Same person (should match)
    print("\n--- Test 1: Same person ---")
    best_confidence = 0.0
    best_student_id = None
    
    for student_id, encodings in student_encodings.items():
        student_max_similarity = 0.0
        for encoding in encodings:
            is_same, similarity = embedder.compare_faces(embedding1, encoding, 0.5)
            if similarity > student_max_similarity:
                student_max_similarity = similarity
    
        if student_max_similarity > 0.95 and student_max_similarity > best_confidence:
            best_confidence = student_max_similarity
            best_student_id = student_id
    
    if best_confidence > 0.95:
        print(f"✅ SAME PERSON MATCH: {best_student_id} with confidence {best_confidence:.4f}")
    else:
        print(f"❌ SAME PERSON FAILED: Best confidence {best_confidence:.4f}")
    
    # Test 2: Different person (should NOT match)
    print("\n--- Test 2: Different person ---")
    best_confidence = 0.0
    best_student_id = None
    
    for student_id, encodings in student_encodings.items():
        student_max_similarity = 0.0
        for encoding in encodings:
            is_same, similarity = embedder.compare_faces(embedding2, encoding, 0.5)
            if similarity > student_max_similarity:
                student_max_similarity = similarity
    
        if student_max_similarity > 0.95 and student_max_similarity > best_confidence:
            best_confidence = student_max_similarity
            best_student_id = student_id
    
    if best_confidence > 0.95:
        print(f"❌ DIFFERENT PERSON INCORRECTLY MATCHED: {best_student_id} with confidence {best_confidence:.4f}")
    else:
        print(f"✅ DIFFERENT PERSON CORRECTLY REJECTED: Best confidence {best_confidence:.4f}")
    
    print("\nDiscrimination fix test completed!")

if __name__ == "__main__":
    test_discrimination_fix()
