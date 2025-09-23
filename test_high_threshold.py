"""
Test the high threshold fix
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def create_realistic_face_image(seed: int, size: int = 128) -> np.ndarray:
    """Create a more realistic face-like image for testing"""
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
    
    # Add eyes
    eye_color = (np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(0, 50))
    left_eye_center = (size // 2 - size // 6, size // 2 - size // 8)
    right_eye_center = (size // 2 + size // 6, size // 2 - size // 8)
    cv2.circle(img, left_eye_center, size // 12, eye_color, -1)
    cv2.circle(img, right_eye_center, size // 12, eye_color, -1)
    
    # Add nose
    nose_color = (np.random.randint(100, 150), np.random.randint(100, 150), np.random.randint(100, 150))
    nose_points = np.array([
        [size // 2, size // 2 - size // 8],
        [size // 2 - size // 20, size // 2 + size // 8],
        [size // 2 + size // 20, size // 2 + size // 8]
    ], np.int32)
    cv2.fillPoly(img, [nose_points], nose_color)
    
    # Add mouth
    mouth_color = (np.random.randint(50, 100), np.random.randint(50, 100), np.random.randint(100, 150))
    mouth_center = (size // 2, size // 2 + size // 4)
    mouth_axes = (size // 8, size // 16)
    cv2.ellipse(img, mouth_center, mouth_axes, 0, 0, 180, mouth_color, -1)
    
    # Add some random texture/noise
    noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def test_high_threshold():
    """Test with high threshold to prevent false matches"""
    print("Testing High Threshold Fix...")
    
    embedder = SimpleFaceEmbedder()
    
    # Create test faces
    face1 = create_realistic_face_image(1, 128)  # Person A (registered)
    face2 = create_realistic_face_image(2, 128)  # Person B (different person)
    face3 = create_realistic_face_image(3, 128)  # Person C (different person)
    
    # Get embeddings
    emb1 = embedder.get_embedding(face1)  # Registered person
    emb2 = embedder.get_embedding(face2)  # Different person
    emb3 = embedder.get_embedding(face3)  # Different person
    
    if emb1 is None or emb2 is None or emb3 is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"Embedding 1 (Person A) norm: {np.linalg.norm(emb1):.4f}")
    print(f"Embedding 2 (Person B) norm: {np.linalg.norm(emb2):.4f}")
    print(f"Embedding 3 (Person C) norm: {np.linalg.norm(emb3):.4f}")
    
    # Test same person (should match)
    print("\n=== Testing Same Person (Should Match) ===")
    emb1_again = embedder.get_embedding(face1)
    is_same, similarity = embedder.compare_faces(emb1, emb1_again, 0.5)
    print(f"Person A vs Person A: is_same={is_same}, similarity={similarity:.4f}")
    
    if similarity > 0.98:
        print("✅ CORRECT: Same person has high similarity (>0.98)")
    else:
        print("❌ WRONG: Same person has low similarity")
    
    # Test different people (should NOT match)
    print("\n=== Testing Different People (Should NOT Match) ===")
    
    # Person A vs Person B
    is_same, similarity = embedder.compare_faces(emb1, emb2, 0.5)
    print(f"Person A vs Person B: is_same={is_same}, similarity={similarity:.4f}")
    
    if similarity < 0.98:
        print("✅ CORRECT: Different people have low similarity (<0.98)")
    else:
        print("❌ WRONG: Different people have high similarity")
    
    # Person A vs Person C
    is_same, similarity = embedder.compare_faces(emb1, emb3, 0.5)
    print(f"Person A vs Person C: is_same={is_same}, similarity={similarity:.4f}")
    
    if similarity < 0.98:
        print("✅ CORRECT: Different people have low similarity (<0.98)")
    else:
        print("❌ WRONG: Different people have high similarity")
    
    # Test with 0.98 threshold
    print("\n=== Testing with 0.98 Threshold ===")
    
    # Same person
    is_same, similarity = embedder.compare_faces(emb1, emb1_again, 0.98)
    print(f"Person A vs Person A (0.98 threshold): is_same={is_same}, similarity={similarity:.4f}")
    
    # Different people
    is_same, similarity = embedder.compare_faces(emb1, emb2, 0.98)
    print(f"Person A vs Person B (0.98 threshold): is_same={is_same}, similarity={similarity:.4f}")
    
    is_same, similarity = embedder.compare_faces(emb1, emb3, 0.98)
    print(f"Person A vs Person C (0.98 threshold): is_same={is_same}, similarity={similarity:.4f}")
    
    # Test multiple different faces
    print("\n=== Testing Multiple Different Faces ===")
    similarities = []
    for i in range(4, 10):  # Test faces 4-9
        face_i = create_realistic_face_image(i, 128)
        emb_i = embedder.get_embedding(face_i)
        if emb_i is not None:
            _, sim = embedder.compare_faces(emb1, emb_i, 0.5)
            similarities.append(sim)
            print(f"Person A vs Person {i}: similarity={sim:.4f}")
    
    if similarities:
        max_similarity = np.max(similarities)
        avg_similarity = np.mean(similarities)
        print(f"Maximum similarity with different people: {max_similarity:.4f}")
        print(f"Average similarity with different people: {avg_similarity:.4f}")
        
        if max_similarity < 0.98:
            print("✅ EXCELLENT: All different people have similarity <0.98")
        elif max_similarity < 0.99:
            print("⚠️  WARNING: Some different people have similarity 0.98-0.99")
        else:
            print("❌ CRITICAL: Some different people have similarity >0.99")

if __name__ == "__main__":
    test_high_threshold()
