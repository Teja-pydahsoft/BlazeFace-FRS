"""
Test the improved simple face embedder
"""

import sys
import os
import cv2
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_improved_embedder():
    """Test the improved simple face embedder"""
    try:
        print("Testing Improved Simple Face Embedder...")
        
        # Initialize embedder
        embedder = SimpleFaceEmbedder()
        
        # Create different test images
        test_image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Add some structure to make them more different
        test_image1[20:80, 20:80] = 255  # White square
        test_image2[30:70, 30:70] = 0    # Black square
        
        # Get embeddings
        embedding1 = embedder.get_embedding(test_image1)
        embedding2 = embedder.get_embedding(test_image2)
        
        if embedding1 is not None and embedding2 is not None:
            print(f"✓ Embeddings generated successfully")
            print(f"  - Shape: {embedding1.shape}")
            print(f"  - Type: {embedding1.dtype}")
            print(f"  - Sample values 1: {embedding1[:5]}")
            print(f"  - Sample values 2: {embedding2[:5]}")
            print(f"  - Norm 1: {np.linalg.norm(embedding1):.4f}")
            print(f"  - Norm 2: {np.linalg.norm(embedding2):.4f}")
        else:
            print("✗ Failed to generate embeddings")
            return
        
        # Test consistency - same image should give same embedding
        embedding1_2 = embedder.get_embedding(test_image1)
        if embedding1_2 is not None:
            is_same, similarity = embedder.compare_faces(embedding1, embedding1_2, 0.9)
            print(f"✓ Consistency test: is_same={is_same}, similarity={similarity:.4f}")
            
            if is_same and similarity > 0.99:
                print("✓ Embeddings are consistent")
            else:
                print("✗ Embeddings are not consistent")
        else:
            print("✗ Failed to generate second embedding")
        
        # Test discrimination - different images should give different embeddings
        is_same, similarity = embedder.compare_faces(embedding1, embedding2, 0.9)
        print(f"✓ Discrimination test: is_same={is_same}, similarity={similarity:.4f}")
        
        if not is_same and similarity < 0.9:
            print("✓ Different images correctly identified as different")
        else:
            print("✗ Different images incorrectly identified as same")
        
        # Test with different thresholds
        print("\nTesting with different thresholds:")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            is_same, similarity = embedder.compare_faces(embedding1, embedding2, threshold)
            print(f"  Threshold {threshold:.1f}: is_same={is_same}, similarity={similarity:.4f}")
        
        print("Improved embedder test completed")
        
    except Exception as e:
        print(f"Error in improved embedder test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_embedder()
