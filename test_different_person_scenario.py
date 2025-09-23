"""
Test what happens when a different person appears with high face detection confidence
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_different_person_scenario():
    """Test what happens when different person has high face detection confidence"""
    print("Testing Different Person Scenario...")
    
    embedder = SimpleFaceEmbedder()
    
    # Create two very different faces
    np.random.seed(1)
    face1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # Person A
    np.random.seed(2)
    face2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # Person B
    
    # Get embeddings
    embedding1 = embedder.get_embedding(face1)  # Person A's encoding
    embedding2 = embedder.get_embedding(face2)  # Person B's encoding
    
    print(f"Person A embedding norm: {np.linalg.norm(embedding1):.4f}")
    print(f"Person B embedding norm: {np.linalg.norm(embedding2):.4f}")
    
    # Simulate stored student encodings (Person A is registered)
    student_encodings = {
        "1234": [embedding1, embedding1]  # Person A's encodings stored
    }
    
    print("\n=== SCENARIO: Different Person B appears with high face detection confidence ===")
    print("Face Detection Confidence: 0.96 (HIGH)")
    print("Registered Student: Person A (ID: 1234)")
    print("Person Appearing: Person B (Different person)")
    
    # Test recognition logic
    best_confidence = 0.0
    best_student_id = None
    comparison_threshold = 0.5
    final_threshold = 0.95
    
    print(f"\nUsing comparison threshold: {comparison_threshold:.2f}")
    print(f"Using final threshold: {final_threshold:.2f}")
    
    for student_id, encodings in student_encodings.items():
        student_max_similarity = 0.0
        student_similarities = []
        
        for i, encoding in enumerate(encodings):
            is_same, similarity = embedder.compare_faces(embedding2, encoding, comparison_threshold)
            student_similarities.append(similarity)
            
            if similarity > student_max_similarity:
                student_max_similarity = similarity
        
        print(f"Student {student_id} similarities: {[f'{s:.4f}' for s in student_similarities]}")
        print(f"Student {student_id} max similarity: {student_max_similarity:.4f}")
        
        if student_max_similarity > final_threshold and student_max_similarity > best_confidence:
            best_confidence = student_max_similarity
            best_student_id = student_id
            print(f"  -> New best match: {student_id} with {student_max_similarity:.4f}")
    
    # Final result
    print(f"\nFinal result: best_confidence={best_confidence:.4f}, best_student_id={best_student_id}")
    
    if best_confidence > final_threshold:
        print(f"❌ WRONG: Different person matched as {best_student_id} with confidence {best_confidence:.4f}")
        print("This would be INCORRECT - different person should be rejected!")
    else:
        print(f"✅ CORRECT: Different person rejected (confidence {best_confidence:.4f} < {final_threshold:.2f})")
        print("This is CORRECT - different person properly rejected!")
    
    print("\n=== CONCLUSION ===")
    print("Even with high face detection confidence (0.96), the system should:")
    print("1. Generate face embedding for Person B")
    print("2. Compare with stored Person A encodings")
    print("3. Find low similarity (<0.95)")
    print("4. Reject as 'Unknown Face'")
    print("5. NOT mark attendance")

if __name__ == "__main__":
    test_different_person_scenario()
