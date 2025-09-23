"""
Test with real face images to verify discrimination
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

def test_realistic_faces():
    """Test with more realistic face images"""
    print("Testing with realistic face images...")
    
    embedder = SimpleFaceEmbedder()
    
    # Create two different realistic faces
    face1 = create_realistic_face_image(1, 128)  # Person A
    face2 = create_realistic_face_image(2, 128)  # Person B
    
    print(f"Face 1 shape: {face1.shape}")
    print(f"Face 2 shape: {face2.shape}")
    
    # Save images for inspection
    cv2.imwrite("test_face1.jpg", face1)
    cv2.imwrite("test_face2.jpg", face2)
    print("Saved test faces as test_face1.jpg and test_face2.jpg")
    
    # Get embeddings
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    
    if emb1 is None or emb2 is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"Embedding 1 shape: {emb1.shape}, norm: {np.linalg.norm(emb1):.4f}")
    print(f"Embedding 2 shape: {emb2.shape}, norm: {np.linalg.norm(emb2):.4f}")
    
    # Test consistency (same face)
    emb1_again = embedder.get_embedding(face1)
    is_same, similarity = embedder.compare_faces(emb1, emb1_again, 0.5)
    print(f"Same face consistency: is_same={is_same}, similarity={similarity:.4f}")
    
    # Test discrimination (different faces)
    is_same, similarity = embedder.compare_faces(emb1, emb2, 0.5)
    print(f"Different faces: is_same={is_same}, similarity={similarity:.4f}")
    
    # Test with different thresholds
    print("\nThreshold Analysis:")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    
    for threshold in thresholds:
        is_same, similarity = embedder.compare_faces(emb1, emb2, threshold)
        status = "✅ CORRECT" if not is_same else "❌ WRONG"
        print(f"Threshold {threshold:.2f}: is_same={is_same}, similarity={similarity:.4f} {status}")
    
    # Test with multiple different faces
    print("\nTesting multiple different faces:")
    similarities = []
    for i in range(3, 8):  # Test faces 3-7
        face_i = create_realistic_face_image(i, 128)
        emb_i = embedder.get_embedding(face_i)
        if emb_i is not None:
            _, sim = embedder.compare_faces(emb1, emb_i, 0.5)
            similarities.append(sim)
            print(f"Face 1 vs Face {i}: similarity={sim:.4f}")
    
    if similarities:
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        print(f"Average similarity with different faces: {avg_similarity:.4f}")
        print(f"Maximum similarity with different faces: {max_similarity:.4f}")
        
        if max_similarity < 0.9:
            print("✅ GOOD: Different faces have low similarity")
        else:
            print("❌ BAD: Different faces have high similarity")

if __name__ == "__main__":
    test_realistic_faces()
