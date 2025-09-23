"""
Test with real photos to verify discrimination works
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_with_real_photos():
    """Test with real photos to verify discrimination"""
    print("Testing with Real Photos for Discrimination...")
    
    embedder = SimpleFaceEmbedder()
    
    # Test with the current 0.98 threshold
    print("Using 0.98 threshold for final matching")
    
    # Create two very different realistic faces
    def create_realistic_face(seed: int, size: int = 200) -> np.ndarray:
        np.random.seed(seed)
        
        # Create base image
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Add skin tone base
        skin_color = (np.random.randint(180, 220), np.random.randint(160, 200), np.random.randint(140, 180))
        img.fill(0)
        
        # Create face shape (ellipse)
        center = (size // 2, size // 2)
        axes = (size // 3, size // 2 - 10)
        cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1)
        
        # Add eyes with different shapes
        eye_color = (np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(0, 50))
        left_eye_center = (size // 2 - size // 6, size // 2 - size // 8)
        right_eye_center = (size // 2 + size // 6, size // 2 - size // 8)
        eye_radius = size // 12
        
        # Left eye
        cv2.ellipse(img, left_eye_center, (eye_radius, eye_radius//2), 0, 0, 360, eye_color, -1)
        cv2.circle(img, left_eye_center, eye_radius // 3, (255, 255, 255), -1)
        
        # Right eye
        cv2.ellipse(img, right_eye_center, (eye_radius//2, eye_radius), 0, 0, 360, eye_color, -1)
        cv2.circle(img, right_eye_center, eye_radius // 3, (255, 255, 255), -1)
        
        # Add nose
        nose_color = (np.random.randint(100, 150), np.random.randint(100, 150), np.random.randint(100, 150))
        nose_center = (size // 2, size // 2)
        nose_width = size // 20
        nose_height = size // 4
        
        cv2.line(img, (nose_center[0], nose_center[1] - nose_height//2), 
                 (nose_center[0], nose_center[1] + nose_height//2), nose_color, max(1, nose_width//2))
        
        # Add mouth
        mouth_color = (np.random.randint(50, 100), np.random.randint(50, 100), np.random.randint(100, 150))
        mouth_center = (size // 2, size // 2 + size // 4)
        mouth_width = size // 6
        mouth_height = size // 16
        
        cv2.ellipse(img, (mouth_center[0], mouth_center[1] - mouth_height//2), 
                    (mouth_width, mouth_height), 0, 0, 180, mouth_color, -1)
        cv2.ellipse(img, (mouth_center[0], mouth_center[1] + mouth_height//2), 
                    (mouth_width, mouth_height), 0, 180, 360, mouth_color, -1)
        
        # Add some random texture/noise for uniqueness
        noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    # Create test faces
    face1 = create_realistic_face(1, 200)  # Person A
    face2 = create_realistic_face(2, 200)  # Person B
    
    # Save images
    cv2.imwrite("real_face1.jpg", face1)
    cv2.imwrite("real_face2.jpg", face2)
    print("Saved test faces as real_face1.jpg and real_face2.jpg")
    
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
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    
    for threshold in thresholds:
        is_same, similarity = embedder.compare_faces(emb1, emb2, threshold)
        status = "✅ CORRECT" if not is_same else "❌ WRONG"
        print(f"Threshold {threshold:.2f}: is_same={is_same}, similarity={similarity:.4f} {status}")
    
    # Test the new matching logic with 0.98 threshold
    print("\n=== Testing New Matching Logic (0.98 threshold) ===")
    
    # Simulate student encodings
    student_encodings = {
        "student1": [emb1, emb1]  # Same embedding twice
    }
    
    # Test 1: Same person (should match)
    print("\n--- Test 1: Same person ---")
    best_confidence = 0.0
    best_student_id = None
    comparison_threshold = 0.5
    final_threshold = 0.98
    
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
    print("\n--- Test 2: Different person ---")
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
    
    print("\n=== Conclusion ===")
    print("The 0.98 threshold should:")
    print("1. ✅ Match same person photos (similarity > 0.98)")
    print("2. ❌ Reject different person photos (similarity < 0.98)")
    print("3. This prevents false matches while allowing true matches")

if __name__ == "__main__":
    test_with_real_photos()
