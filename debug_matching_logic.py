"""
Debug the matching logic to find the bug
"""

import sys
import os
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def debug_matching_logic():
    """Debug the matching logic step by step"""
    print("=== Debugging Matching Logic ===")
    
    embedder = SimpleFaceEmbedder()
    
    # Create test data
    np.random.seed(1)
    face1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    face2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    
    # Simulate student encodings
    student_encodings = {
        "123": [emb1, emb1]  # Same embedding twice
    }
    
    # Test the exact logic from attendance system
    print("\n=== Testing Exact Logic ===")
    
    query_embedding = emb1  # Same as stored
    threshold = 0.6  # UI threshold
    comparison_threshold = max(threshold, 0.8)  # Should be 0.8
    
    print(f"UI threshold: {threshold}")
    print(f"Comparison threshold: {comparison_threshold}")
    
    best_confidence = 0.0
    best_student_id = None
    all_similarities = []
    
    for student_id, encodings in student_encodings.items():
        print(f"Checking student {student_id} with {len(encodings)} encodings")
        student_max_similarity = 0.0
        
        for i, encoding in enumerate(encodings):
            is_same, similarity = embedder.compare_faces(query_embedding, encoding, 0.5)
            all_similarities.append((student_id, i, similarity))
            
            if similarity > student_max_similarity:
                student_max_similarity = similarity
            print(f"  Encoding {i}: similarity={similarity:.4f}")
        
        print(f"  Student {student_id} max similarity: {student_max_similarity:.4f}")
        print(f"  Condition 1: {student_max_similarity:.4f} > {comparison_threshold:.4f} = {student_max_similarity > comparison_threshold}")
        print(f"  Condition 2: {student_max_similarity:.4f} > {best_confidence:.4f} = {student_max_similarity > best_confidence}")
        print(f"  Both conditions: {student_max_similarity > comparison_threshold and student_max_similarity > best_confidence}")
        
        if student_max_similarity > comparison_threshold and student_max_similarity > best_confidence:
            best_confidence = student_max_similarity
            best_student_id = student_id
            print(f"  -> New best match: {student_id} with {student_max_similarity:.4f}")
        else:
            print(f"  -> No match (conditions not met)")
    
    print(f"\nFinal result: best_confidence={best_confidence:.4f}, best_student_id={best_student_id}")
    
    # Test with different face
    print(f"\n=== Testing with Different Face ===")
    
    query_embedding = emb2  # Different face
    best_confidence = 0.0
    best_student_id = None
    
    for student_id, encodings in student_encodings.items():
        print(f"Checking student {student_id} with {len(encodings)} encodings")
        student_max_similarity = 0.0
        
        for i, encoding in enumerate(encodings):
            is_same, similarity = embedder.compare_faces(query_embedding, encoding, 0.5)
            
            if similarity > student_max_similarity:
                student_max_similarity = similarity
            print(f"  Encoding {i}: similarity={similarity:.4f}")
        
        print(f"  Student {student_id} max similarity: {student_max_similarity:.4f}")
        print(f"  Condition 1: {student_max_similarity:.4f} > {comparison_threshold:.4f} = {student_max_similarity > comparison_threshold}")
        print(f"  Condition 2: {student_max_similarity:.4f} > {best_confidence:.4f} = {student_max_similarity > best_confidence}")
        print(f"  Both conditions: {student_max_similarity > comparison_threshold and student_max_similarity > best_confidence}")
        
        if student_max_similarity > comparison_threshold and student_max_similarity > best_confidence:
            best_confidence = student_max_similarity
            best_student_id = student_id
            print(f"  -> New best match: {student_id} with {student_max_similarity:.4f}")
        else:
            print(f"  -> No match (conditions not met)")
    
    print(f"\nFinal result: best_confidence={best_confidence:.4f}, best_student_id={best_student_id}")

if __name__ == "__main__":
    debug_matching_logic()
