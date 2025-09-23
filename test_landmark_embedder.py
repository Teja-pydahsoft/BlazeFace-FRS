"""
Test landmark-based facial feature recognition
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.landmark_face_embedder import LandmarkFaceEmbedder

def create_detailed_face_image(seed: int, size: int = 200) -> np.ndarray:
    """Create a detailed face image with distinct facial features"""
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
    
    # Add distinct eyes with different shapes
    eye_color = (np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(0, 50))
    left_eye_center = (size // 2 - size // 6, size // 2 - size // 8)
    right_eye_center = (size // 2 + size // 6, size // 2 - size // 8)
    eye_radius = size // 12
    
    # Left eye (different shape)
    cv2.ellipse(img, left_eye_center, (eye_radius, eye_radius//2), 0, 0, 360, eye_color, -1)
    cv2.circle(img, left_eye_center, eye_radius // 3, (255, 255, 255), -1)  # Eye highlight
    
    # Right eye (different shape)
    cv2.ellipse(img, right_eye_center, (eye_radius//2, eye_radius), 0, 0, 360, eye_color, -1)
    cv2.circle(img, right_eye_center, eye_radius // 3, (255, 255, 255), -1)  # Eye highlight
    
    # Add distinct nose with different shapes
    nose_color = (np.random.randint(100, 150), np.random.randint(100, 150), np.random.randint(100, 150))
    nose_center = (size // 2, size // 2)
    nose_width = size // 20
    nose_height = size // 4
    
    # Nose bridge (different width)
    cv2.line(img, (nose_center[0], nose_center[1] - nose_height//2), 
             (nose_center[0], nose_center[1] + nose_height//2), nose_color, max(1, nose_width//2))
    
    # Nose nostrils (different sizes)
    cv2.circle(img, (nose_center[0] - nose_width, nose_center[1] + nose_height//3), 
               nose_width//2, nose_color, -1)
    cv2.circle(img, (nose_center[0] + nose_width, nose_center[1] + nose_height//3), 
               nose_width//2, nose_color, -1)
    
    # Add distinct mouth with different shapes
    mouth_color = (np.random.randint(50, 100), np.random.randint(50, 100), np.random.randint(100, 150))
    mouth_center = (size // 2, size // 2 + size // 4)
    mouth_width = size // 6
    mouth_height = size // 16
    
    # Upper lip (different curve)
    cv2.ellipse(img, (mouth_center[0], mouth_center[1] - mouth_height//2), 
                (mouth_width, mouth_height), 0, 0, 180, mouth_color, -1)
    
    # Lower lip (different curve)
    cv2.ellipse(img, (mouth_center[0], mouth_center[1] + mouth_height//2), 
                (mouth_width, mouth_height), 0, 180, 360, mouth_color, -1)
    
    # Add distinct jawline with different shapes
    jawline_color = (np.random.randint(120, 160), np.random.randint(120, 160), np.random.randint(120, 160))
    jawline_points = np.array([
        [size // 2 - size // 3, size // 2 + size // 3],
        [size // 2 - size // 4, size // 2 + size // 2],
        [size // 2, size // 2 + size // 2 + size // 8],
        [size // 2 + size // 4, size // 2 + size // 2],
        [size // 2 + size // 3, size // 2 + size // 3]
    ], np.int32)
    cv2.polylines(img, [jawline_points], False, jawline_color, 2)
    
    # Add distinct eyebrows with different shapes
    eyebrow_color = (np.random.randint(80, 120), np.random.randint(80, 120), np.random.randint(80, 120))
    left_eyebrow = np.array([
        [size // 2 - size // 4, size // 2 - size // 6],
        [size // 2 - size // 8, size // 2 - size // 8],
        [size // 2 - size // 12, size // 2 - size // 6]
    ], np.int32)
    right_eyebrow = np.array([
        [size // 2 + size // 12, size // 2 - size // 6],
        [size // 2 + size // 8, size // 2 - size // 8],
        [size // 2 + size // 4, size // 2 - size // 6]
    ], np.int32)
    cv2.polylines(img, [left_eyebrow], False, eyebrow_color, 2)
    cv2.polylines(img, [right_eyebrow], False, eyebrow_color, 2)
    
    # Add some random texture/noise for uniqueness
    noise = np.random.randint(-10, 10, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def test_landmark_embedder():
    """Test landmark-based facial feature recognition"""
    print("Testing Landmark-Based Facial Feature Recognition...")
    
    try:
        embedder = LandmarkFaceEmbedder()
    except Exception as e:
        print(f"❌ Failed to initialize LandmarkFaceEmbedder: {e}")
        return
    
    # Create detailed face images
    face1 = create_detailed_face_image(1, 200)  # Person A
    face2 = create_detailed_face_image(2, 200)  # Person B
    face3 = create_detailed_face_image(3, 200)  # Person C
    
    # Save images for inspection
    cv2.imwrite("landmark_face1.jpg", face1)
    cv2.imwrite("landmark_face2.jpg", face2)
    cv2.imwrite("landmark_face3.jpg", face3)
    print("Saved landmark faces as landmark_face1.jpg, landmark_face2.jpg, landmark_face3.jpg")
    
    # Get embeddings
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    emb3 = embedder.get_embedding(face3)
    
    if emb1 is None or emb2 is None or emb3 is None:
        print("❌ Failed to generate landmark embeddings")
        return
    
    print(f"Landmark embedding 1 shape: {emb1.shape}, norm: {np.linalg.norm(emb1):.4f}")
    print(f"Landmark embedding 2 shape: {emb2.shape}, norm: {np.linalg.norm(emb2):.4f}")
    print(f"Landmark embedding 3 shape: {emb3.shape}, norm: {np.linalg.norm(emb3):.4f}")
    
    # Test same person (should match)
    print("\n=== Testing Same Person (Should Match) ===")
    emb1_again = embedder.get_embedding(face1)
    is_same, similarity = embedder.compare_faces(emb1, emb1_again, 0.5)
    print(f"Person A vs Person A: is_same={is_same}, similarity={similarity:.4f}")
    
    if similarity > 0.95:
        print("✅ EXCELLENT: Same person has very high similarity (>0.95)")
    elif similarity > 0.9:
        print("✅ GOOD: Same person has high similarity (>0.9)")
    else:
        print("❌ POOR: Same person has low similarity")
    
    # Test different people (should NOT match)
    print("\n=== Testing Different People (Should NOT Match) ===")
    
    # Person A vs Person B
    is_same, similarity = embedder.compare_faces(emb1, emb2, 0.5)
    print(f"Person A vs Person B: is_same={is_same}, similarity={similarity:.4f}")
    
    if similarity < 0.8:
        print("✅ EXCELLENT: Different people have very low similarity (<0.8)")
    elif similarity < 0.9:
        print("✅ GOOD: Different people have low similarity (<0.9)")
    else:
        print("❌ POOR: Different people have high similarity")
    
    # Person A vs Person C
    is_same, similarity = embedder.compare_faces(emb1, emb3, 0.5)
    print(f"Person A vs Person C: is_same={is_same}, similarity={similarity:.4f}")
    
    if similarity < 0.8:
        print("✅ EXCELLENT: Different people have very low similarity (<0.8)")
    elif similarity < 0.9:
        print("✅ GOOD: Different people have low similarity (<0.9)")
    else:
        print("❌ POOR: Different people have high similarity")
    
    # Test with high threshold
    print("\n=== Testing with High Threshold (0.95) ===")
    
    # Same person
    is_same, similarity = embedder.compare_faces(emb1, emb1_again, 0.95)
    print(f"Person A vs Person A (0.95 threshold): is_same={is_same}, similarity={similarity:.4f}")
    
    # Different people
    is_same, similarity = embedder.compare_faces(emb1, emb2, 0.95)
    print(f"Person A vs Person B (0.95 threshold): is_same={is_same}, similarity={similarity:.4f}")
    
    is_same, similarity = embedder.compare_faces(emb1, emb3, 0.95)
    print(f"Person A vs Person C (0.95 threshold): is_same={is_same}, similarity={similarity:.4f}")
    
    # Test multiple different faces
    print("\n=== Testing Multiple Different Faces ===")
    similarities = []
    for i in range(4, 8):  # Test faces 4-7
        face_i = create_detailed_face_image(i, 200)
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
        
        if max_similarity < 0.8:
            print("✅ EXCELLENT: All different people have similarity <0.8")
        elif max_similarity < 0.9:
            print("✅ GOOD: All different people have similarity <0.9")
        else:
            print("❌ POOR: Some different people have similarity >0.9")
    
    print("\n=== Landmark Feature Analysis ===")
    print("This embedder extracts:")
    print("- Eye landmark features (shape, position, depth)")
    print("- Nose landmark features (shape, size, position)")
    print("- Mouth landmark features (shape, size, position)")
    print("- Jawline landmark features (contour, shape)")
    print("- Eyebrow landmark features (shape, position)")
    print("- Face shape features (dimensions, aspect ratios)")
    print("- Uses MediaPipe for precise facial landmark detection")

if __name__ == "__main__":
    test_landmark_embedder()
