"""
Test discrimination with more realistic face-like images
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def create_face_like_image(seed, pattern_type="circles"):
    """Create a more realistic face-like image"""
    np.random.seed(seed)
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    
    if pattern_type == "circles":
        # Create circular patterns (like eyes, nose, mouth)
        cv2.circle(img, (40, 50), 15, (100, 100, 100), -1)  # Left eye
        cv2.circle(img, (120, 50), 15, (100, 100, 100), -1)  # Right eye
        cv2.circle(img, (80, 80), 10, (80, 80, 80), -1)  # Nose
        cv2.ellipse(img, (80, 110), (20, 10), 0, 0, 180, (60, 60, 60), -1)  # Mouth
    elif pattern_type == "squares":
        # Create square patterns
        cv2.rectangle(img, (30, 40), (50, 60), (120, 120, 120), -1)  # Left eye
        cv2.rectangle(img, (110, 40), (130, 60), (120, 120, 120), -1)  # Right eye
        cv2.rectangle(img, (70, 70), (90, 90), (100, 100, 100), -1)  # Nose
        cv2.rectangle(img, (60, 100), (100, 120), (80, 80, 80), -1)  # Mouth
    elif pattern_type == "lines":
        # Create line patterns
        cv2.line(img, (30, 50), (50, 50), (150, 150, 150), 3)  # Left eye
        cv2.line(img, (110, 50), (130, 50), (150, 150, 150), 3)  # Right eye
        cv2.line(img, (80, 70), (80, 90), (130, 130, 130), 3)  # Nose
        cv2.line(img, (60, 110), (100, 110), (110, 110, 110), 3)  # Mouth
    
    # Add some noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def test_realistic_discrimination():
    """Test discrimination with more realistic face-like images"""
    try:
        print("Testing Realistic Face Discrimination...")
        
        embedder = SimpleFaceEmbedder()
        
        # Create three different face-like images
        face1 = create_face_like_image(1, "circles")
        face2 = create_face_like_image(2, "squares")
        face3 = create_face_like_image(3, "lines")
        
        # Get embeddings
        emb1 = embedder.get_embedding(face1)
        emb2 = embedder.get_embedding(face2)
        emb3 = embedder.get_embedding(face3)
        
        if emb1 is None or emb2 is None or emb3 is None:
            print("✗ Failed to generate embeddings")
            return
        
        print("✓ Embeddings generated successfully")
        print(f"  Embedding 1 shape: {emb1.shape}")
        print(f"  Embedding 2 shape: {emb2.shape}")
        print(f"  Embedding 3 shape: {emb3.shape}")
        
        # Test consistency (same face should match)
        is_same_1_1, sim_1_1 = embedder.compare_faces(emb1, emb1, 0.8)
        print(f"Face1 vs Face1: is_same={is_same_1_1}, similarity={sim_1_1:.4f}")
        
        # Test discrimination (different faces should not match)
        is_same_1_2, sim_1_2 = embedder.compare_faces(emb1, emb2, 0.8)
        is_same_1_3, sim_1_3 = embedder.compare_faces(emb1, emb3, 0.8)
        is_same_2_3, sim_2_3 = embedder.compare_faces(emb2, emb3, 0.8)
        
        print(f"Face1 vs Face2: is_same={is_same_1_2}, similarity={sim_1_2:.4f}")
        print(f"Face1 vs Face3: is_same={is_same_1_3}, similarity={sim_1_3:.4f}")
        print(f"Face2 vs Face3: is_same={is_same_2_3}, similarity={sim_2_3:.4f}")
        
        # Check results
        if is_same_1_1 and not is_same_1_2 and not is_same_1_3 and not is_same_2_3:
            print("✓ Discrimination test PASSED - different faces correctly identified as different")
        else:
            print("✗ Discrimination test FAILED - some different faces incorrectly identified as same")
        
        # Test with very high threshold
        print("\nTesting with very high threshold (0.95):")
        is_same_high, sim_high = embedder.compare_faces(emb1, emb2, 0.95)
        print(f"  Threshold 0.95: is_same={is_same_high}, similarity={sim_high:.4f}")
        
    except Exception as e:
        print(f"Error in realistic discrimination test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_realistic_discrimination()
