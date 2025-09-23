"""
Test improved face discrimination
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_discrimination():
    """Test face discrimination with different faces"""
    try:
        print("Testing Improved Face Discrimination...")
        
        embedder = SimpleFaceEmbedder()
        
        # Create three different dummy faces
        np.random.seed(1)
        face1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        np.random.seed(2)
        face2 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        np.random.seed(3)
        face3 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        # Get embeddings
        emb1 = embedder.get_embedding(face1)
        emb2 = embedder.get_embedding(face2)
        emb3 = embedder.get_embedding(face3)
        
        if emb1 is None or emb2 is None or emb3 is None:
            print("✗ Failed to generate embeddings")
            return
        
        print("✓ Embeddings generated successfully")
        
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
        
        # Test with different thresholds
        print("\nTesting with different thresholds:")
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in thresholds:
            is_same, sim = embedder.compare_faces(emb1, emb2, threshold)
            print(f"  Threshold {threshold:.1f}: is_same={is_same}, similarity={sim:.4f}")
        
    except Exception as e:
        print(f"Error in discrimination test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_discrimination()
